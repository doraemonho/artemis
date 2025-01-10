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
//! \brief Initializes a vertical shear instability in spherical coordinates.
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
	Real dslope, pslope;  // density and pressure slopes
	Real rho0, r0, hg0;   // reference density, radius, and gas scale height
	Real rexp;            // radial exponent for density profile
	Real amp;             // amplitude of perturbation
	Real dust_to_gas;     // dust to gas ratio

};


//----------------------------------------------------------------------------------------
//! \fn void InitStratParams
//! \brief Extracts strat parameters from ParameterInput.
//! NOTE(@pdmullen): In order for our user-defined BCs to be compatible with restarts,
//! we must reset the StratParams struct upon initialization.
inline void InitStratParams(MeshBlock *pmb, ParameterInput *pin) {

	Params &params = pmb->packages.Get("artemis")->AllParams();
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto &grav_pkg = pmb->packages.Get("gravity");
  auto &gas_pkg = pmb->packages.Get("gas");
	
	if (!(params.hasKey("VSI_params"))) {
		VSI_Params vsi_params;

		// Get the gas related parameters from the input file
		vsi_params.gm = grav_pkg->Param<Real>("gm");
		vsi_params.r0 = pin->GetReal("problem", "r0");

		// Get the dust related parameters if dust is enabled
		/*const bool do_dust = artemis_pkg->Param<bool>("do_dust");
		if (do_dust) {

		}*/
		params.Add("VSI_params", vsi_params);
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
	auto vsi_params = artemis_pkg->Param<DiskParams>("vsi_params");

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

		auto [r, theta, phi] = x_sph;
		auto [R, a, z] = x_cyl;

		// gas variables
		Real gdens = Null<Real>(), gpres = Null<Real>(), cs2 = Null<Real>();
		Real gvel1 = Null<Real>(), gvel2 = Null<Real>(), gvel3 = Null<Real>();
		
		cs2 = Cs2Profile(vsip, eos_d, r, R, z);
		gpres = gdens*cs2/vsip.gamma;
		gdens = DensityProfile_Gas(vsip, eos_d, r, R, z);

		Real lambda0 = 0.0, lambda1 = 0.0;
		GasVelProfileCyl(vsip, eos_d, 
								    	gdens, cs2, 
								    	lambda0, lambda1,
								    	r, R, z,
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
		v(0, gas::prim::sie(0), k, j, i) = eos_d.InternalEnergyFromDensityPressure(gdens, gpres);
		
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
//! \brief Computes density profile at spherical r and cylindrical R, z
KOKKOS_INLINE_FUNCTION
Real DensityProfile_Gas(VSIParams pgen, EOS eos, 
								const Real r, const Real R, const Real z) {
  
	Real Hgas = pgen.hg0 * std::pow(r/pgen.r0, (pgen.pslope + 3)/2);  
	Real denmid = pgen.rho0 * std::pow(r / pgen.r0, pgen.dslope);
	if (pgen.rexp > 0)
		denmid*=exp(-r/pgen.rexp);
	Real dentem = denmid * std::exp(-SQR(R / Hgas) * (R / r - 1.0));

	return dentem;
}

// funciton to compute the cs^2
//----------------------------------------------------------------------------------------
//! \fn Real Cs2Profile
//! \brief Computes cs^2 profile at spherical r and cylindrical R, z
KOKKOS_INLINE_FUNCTION
Real Cs2Profile(VSIParams pgen, EOS eos, 
							const Real r, const Real R, const Real z) {
	Real poverr = std::(pgen.hg0,2)*(pgen.gm/pgen.r0/pgen.r0/pgen.r0)*
							  std::pow(r/pgen.r0, pgen.pslope);
	return poverr;
}

//----------------------------------------------------------------------------------------
//! \fn Real Cs2Profile
//! \brief Computes cs^2 profile at spherical r and cylindrical R, z
KOKKOS_INLINE_FUNCTION
void GasVelProfileCyl(VSIParams pgen, EOS eos, 
							const Real rhog, const Real cs2, 
							const Real lambda0, const Real lambda1,
							const Real r, const Real R, const Real z,
							Real &v1, Real &v2, Real &v3) {
  // P = rho*cs^2 -> dP/dr = d(rho)/dr*cs^2 + rho*dcs2/dr
  // cs2 = p_over_r*std::pow(rad/r0, pslope);
  // rho(R) ~ rho0*(R/r0)^dslope*exp(-z^2/(2*H^2))
	Real p_over_r = std::(pgen.hg0,2)*(pgen.gm/pgen.r0/pgen.r0/pgen.r0);
	// TODO: Check this
	Real drhodr   = pgen.dslope/pgen.r0*std::pow(r/pgen.r0,-1.0)*rhog;
	if (pgen.rexp > 0)
		drhodr += -rhog/pgen.rexp;

	Real dcs2dr  = pgen.pslope/pgen.r0*p_over_r*std::pow(r/pgen.r0, pgen.pslope - 1);
	Real dpdr    = drhodr*cs2 + rhog*dcs2dr;
	Real omega_k = std::sqrt(pgen.gm/(r*r*r));
	Real vk      = omega_k*r;
	Real vp      = 1/rhog/omega_k*dpdr;
	Real visc    = 0.0; // it is zeros now.

	Real delta_gas_vr   = Delta_gas_vr(vp, visc, lambda0, lambda1);
	Real delta_gas_vphi = Delta_gas_vphi(vp, visc, lambda0, lambda1);

	v1 = delta_gas_vr;
	v2 = 0.0;
	v3 = delta_gas_vphi + vk;

	return ;
}

//----------------------------------------------------------------------------------------
//! \fn Real Delta_gas_vr
//! \brief V_r Equation 11 from  Dipierro+18 MNRAS 479, 4187–4206 (2018)
KOKKOS_INLINE_FUNCTION
Real Delta_gas_vr(const Real vp, const Real visc, const Real lambda0, const Real lambda1) {
  Real dv_g_r = (-lambda1*vp + (1 + lambda0)*visc)/(std::pow(1 + lambda0,2.0) + lambda1*lambda1);
  return dv_g_r;
}

//----------------------------------------------------------------------------------------
//! \fn Real Delta_gas_vphi
//! \brief V_phi  Equation 12 from  Dipierro+18 MNRAS 479, 4187–4206 (2018)
//!        have to add back the Keplerian velocity
KOKKOS_INLINE_FUNCTION
Real Delta_gas_vphi(const Real vp, const Real visc, const Real lambda0, const Real lambda1){
  Real dv_g_phi = 0.5*(vp*(1 + lambda0) + lambda1*visc)/(std::pow(1 + lambda0,2.0) + lambda1*lambda1);
  return dv_g_phi;
}

} // namespace VSI
#endif // PGEN_VSI_HPP_