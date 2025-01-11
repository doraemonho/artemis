//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

//! \file VSI.hpp
//! \brief (KW HO) Initializes a vertical shear instability in spherical coordinates.
#ifndef PGEN_VSI_HPP_
#define PGEN_VSI_HPP_

// C/C++ headers
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>


// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;


namespace VSI {

// contains parameters for VSI problem
struct VSI_Params {
	Real gm;              // gravitational constant
	Real gamma;           // adiabatic index
	Real dslope, pslope;  // density and cs2 slopes
	Real r0; 				   		// reference radius
	Real rho0, hg0;       // reference density, and gas scale height at r0
	Real rexp;            // radial exponent for density profile
	Real amp;             // amplitude of perturbation
	Real dust_to_gas;     // dust to gas ratio
};


//----------------------------------------------------------------------------------------
//! \fn void InitStratParams
//! \brief Extracts strat parameters from ParameterInput.
//! NOTE(@pdmullen): In order for our user-defined BCs to be compatible with restarts,
//! we must reset the StratParams struct upon initialization.
inline void InitVSIParams(MeshBlock *pmb, ParameterInput *pin) {

	Params &params = pmb->packages.Get("artemis")->AllParams();
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto &grav_pkg = pmb->packages.Get("gravity");
  auto &gas_pkg = pmb->packages.Get("gas");
	
	if (!(params.hasKey("VSI_params"))) {
		VSI_Params vsi_params;

		// Get the gas related parameters from the input file
		vsi_params.gm = grav_pkg->Param<Real>("gm");
		vsi_params.r0 = pin->GetReal("problem", "r0");
		vsi_params.amp = pin->GetReal("problem", "amp");
		vsi_params.rho0 = pin->GetReal("problem", "rho0");
		vsi_params.hg0 = pin->GetReal("problem", "hg0");
		vsi_params.rexp = pin->GetOrAddReal("problem", "rexp",0.0);

		vsi_params.gamma = gas_pkg->Param<Real>("adiabatic_index");
		vsi_params.dslope = pin->GetReal("problem", "dslope");
		vsi_params.pslope = pin->GetReal("problem", "pslope");

		// Get the dust related parameters if dust is enabled
		/*const bool do_dust = artemis_pkg->Param<bool>("do_dust");
		if (do_dust) {
			vsi_params.dust_to_gas = pin->GetReal("problem", "dust_to_gas");
		}*/
		params.Add("vsi_params", vsi_params);
		// print out all the parameters
		//if (parthenon::Globals::my_rank == 0) {
		//	std::cout << "VSI Parameters:" << std::endl;
		//	std::cout << "gm: " << vsi_params.gm << std::endl;
		//	std::cout << "r0: " << vsi_params.r0 << std::endl;
		//	std::cout << "rho0: " << vsi_params.rho0 << std::endl;
		//	std::cout << "hg0: " << vsi_params.hg0 << std::endl;
		//	std::cout << "rexp: " << vsi_params.rexp << std::endl;
		//	std::cout << "amp: " << vsi_params.amp << std::endl;
		//	std::cout << "gamma: " << vsi_params.gamma << std::endl;
		//	std::cout << "dslope: " << vsi_params.dslope << std::endl;
		//	std::cout << "pslope: " << vsi_params.pslope << std::endl;
		//	
		//}
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::strat()
//! \brief Sets initial conditions for SI-shearing box problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
	using parthenon::MakePackDescriptor;

  auto artemis_pkg = pmb->packages.Get("artemis");
	const bool do_gas = artemis_pkg->Param<bool>("do_gas");
	// check the pgen requirements
	PARTHENON_REQUIRE(do_gas, "VSI pgen requires gas=true!");
	PARTHENON_REQUIRE( (GEOM == Coordinates::spherical3D) || (GEOM == Coordinates::spherical2D),
								    	"problem = VSI only works for Spherical Coordinates!");

  // Packing
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);


  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
	auto vsi_params = artemis_pkg->Param<VSI_Params>("vsi_params");

  auto &pco = pmb->coords;
  auto &vsip = vsi_params;

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/pmb->gid);

	// Set up the gas initial conditions
	pmb->par_for(
		"VSI_gas", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
		KOKKOS_LAMBDA(const int k, const int j, const int i) {
		// acquire the state of the random number generator engine
		auto generator = random_pool.get_state();

		geometry::Coords<GEOM> coords(pco, k, j, i);
		const auto &x_sph = coords.GetCellCenter();
	  const auto &x_cyl = coords.ConvertCoordsToCyl(x_sph);

		auto [r_sph, t_sph, p_sph] = x_sph;
		auto [r_cyl, p_cyl, z_cyl] = x_cyl;
		auto r_cyl = r*std::sin(theta);
		auto z_cyl = r*std::cos(theta);

		// gas variables
		Real gdens = Null<Real>(), gpres = Null<Real>(), cs2 = Null<Real>();
		Real gvel1 = Null<Real>(), gvel2 = Null<Real>(), gvel3 = Null<Real>();
		Real gtemp = Null<Real>();

		cs2 = Cs2Profile(vsip, eos_d, r_cyl, z_cyl);
		gdens = DensityProfile_Gas(vsip, eos_d, r_cyl, z_cyl);
		gpres = gdens*cs2/vsip.gamma;
		gtemp = gpres/gdens/(vsip.gamma - 1.0);

		Real lambda0 = 0.0, lambda1 = 0.0;
		GasVelProfileCyl(vsip, eos_d, 
								    	gdens, cs2, 
								    	lambda0, lambda1,
								    	r_cyl, z_cyl,
								    	gvel1, gvel2, gvel3);

		const Real del_vx1 =
			vsip.amp * std::sqrt(cs2) * generator.drand(-0.5, 0.5); // ran(iseed);
		const Real del_vx2 =
			vsip.amp * std::sqrt(cs2) * generator.drand(-0.5, 0.5); // ran(iseed);
		const Real del_vx3 =
			vsip.amp * std::sqrt(cs2) * generator.drand(-0.5, 0.5); // ran(iseed);

		v(0, gas::prim::density(0), k, j, i) = gdens;
		v(0, gas::prim::velocity(0), k, j, i) = gvel1 + del_vx1;
		v(0, gas::prim::velocity(1), k, j, i) = gvel2 + del_vx2;
		v(0, gas::prim::velocity(2), k, j, i) = gvel3 + del_vx3;
		v(0, gas::prim::sie(0), k, j, i) = eos_d.InternalEnergyFromDensityTemperature(gdens, gtemp);
		
		// do not forget to release the state of the engine
		random_pool.free_state(generator);
	});

	// Set up the dust initial conditions
	/*if (do_dust) {
		// Set the up dust initial conditions
		pmb->par_for(
			"VSI_dust", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
			KOKKOS_LAMBDA(const int k, const int j, const int i) {

			geometry::Coords<GEOM> coords(pco, k, j, i);

		}
	}*/
}

//----------------------------------------------------------------------------------------
//! \fn Real DensityProfile_Gas
//! \brief Computes density profile at spherical r, z.
//         Modified from A&A666, A98 (2022).
//         rho(r,z) = rho0*(r/r0)^P*exp(-r^2/H^2*(r/R - 1))
KOKKOS_INLINE_FUNCTION
Real DensityProfile_Gas(struct VSI_Params pgen, EOS eos, 
								        const Real r, const Real z) {
  
	Real Hgas = pgen.hg0 * std::pow(r / pgen.r0, (pgen.pslope + 3)/2);  
	Real denmid = pgen.rho0 * std::pow(r / pgen.r0, pgen.dslope);
	if (pgen.rexp > 0)
		denmid*=exp(-r/pgen.rexp);
	Real  r_sph = std::sqrt(r * r + z * z);
	Real dentem = denmid * std::exp(-SQR(r_sph / Hgas) * (r_sph / r - 1.0));

	return dentem;
}

// funciton to compute the cs^2
//----------------------------------------------------------------------------------------
//! \fn Real Cs2Profile
//! \brief Computes cs^2 profile at cylindrical r, z
KOKKOS_INLINE_FUNCTION
Real Cs2Profile(struct VSI_Params pgen, EOS eos, 
						    const Real r, const Real z) {
	Real poverr = std::pow(pgen.hg0,2)*(pgen.gm/pgen.r0/pgen.r0/pgen.r0)*
							  std::pow(r/pgen.r0, pgen.pslope);
	return poverr;
}

//----------------------------------------------------------------------------------------
//! \fn Real Cs2Profile
//! \brief Computes cs^2 profile at cylindrical R, z
KOKKOS_INLINE_FUNCTION
void GasVelProfileCyl(struct VSI_Params pgen, EOS eos, 
											const Real rhog, const Real cs2, 
											const Real lambda0, const Real lambda1,
											const Real r, const Real z,
											Real &v1, Real &v2, Real &v3) {
  // P = rho*cs^2 -> dP/dr = d(rho)/dr*cs^2 + rho*dcs2/dr
  // cs2 = p_over_r*std::pow(rad/r0, pslope);
  // rho(R) ~ rho0*(R/r0)^dslope*exp(-z^2/(2*H^2))
	Real p_over_r = std::pow(pgen.hg0,2)*(pgen.gm/pgen.r0/pgen.r0/pgen.r0);
	// TODO: Check this
	Real drhodr   = pgen.dslope/pgen.r0*std::pow(r/pgen.r0,-1.0)*rhog;
	if (pgen.rexp > 0)
		drhodr += -rhog/pgen.rexp;

	Real dcs2dr  = pgen.pslope/pgen.r0*p_over_r*std::pow(r/pgen.r0, pgen.pslope - 1);
	Real dpdr    = drhodr*cs2 + rhog*dcs2dr;
	Real r_sph   = std::sqrt(r * r + z * z);
	Real omega_k = std::sqrt(pgen.gm/(r_sph*r_sph*r_sph));
	Real vk      = omega_k*r_sph;
	Real vp      = 1/rhog/omega_k*dpdr;
	Real visc    = 0.0; // (TODO):it is zeros now.

	// Equation 11 from  Dipierro+18 MNRAS 479, 4187–4206 (2018)
	Real delta_gas_vr   = (-lambda1*vp + (1 + lambda0)*visc)/
	                      (std::pow(1 + lambda0,2.0) + lambda1*lambda1);

	// Equation 12 from  Dipierro+18 MNRAS 479, 4187–4206 (2018), have to add back the Keplerian velocity
	Real delta_gas_vphi = 0.5*(vp*(1 + lambda0) + lambda1*visc)/
	                      (std::pow(1 + lambda0,2.0) + lambda1*lambda1);

	v1 = delta_gas_vr;
	v2 = 0.0;
	v3 = delta_gas_vphi + vk;

	return ;
}

} // namespace VSI
#endif // PGEN_VSI_HPP_