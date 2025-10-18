/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Drude.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

Drude::Drude (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_species_names.size() == 1,
        "Background stopping must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);
}

void
Drude::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("Drude::doCollisions()");
    using namespace amrex::literals;

    auto& species = mypc->GetParticleContainerFromName(m_species_names[0]);
    amrex::ParticleReal const species_mass = species.getMass();
    amrex::ParticleReal const species_charge = species.getCharge();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_mass > 0_prt, "Error: With background stopping, the species mass must be > 0");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_charge != 0_prt, "Error: With background stopping, the species charge must be nonzero");

    // Loop over refinement levels
    auto const flvl = species.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // loop over particles box by box
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species, lev); pti.isValid(); ++pti) {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doDrudeWithinTile(pti, dt, cur_time, species_mass, species_charge);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add(&(*cost)[pti.index()], wt);
            }
        }

    }
}

void Drude::doDrudeWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                               amrex::ParticleReal species_mass, amrex::ParticleReal species_charge)
{
    using namespace amrex::literals;

    // So that CUDA code gets its intrinsic, not the host-only C++ library version
    using std::sqrt, std::abs, std::log, std::exp;

    // get particle count
    long const np = pti.numParticles();

    // get Struct-Of-Array particle data, also called attribs
    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

    amrex::ParallelFor(np,
        [=] AMREX_GPU_HOST_DEVICE (long ip)
        {

            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
            constexpr amrex::ParticleReal nu = 2e14;
            const amrex::ParticleReal u2 = ux[ip]*ux[ip] + uy[ip]*uy[ip] + uz[ip]*uz[ip];
            const amrex::ParticleReal gamma = sqrt(1._prt + u2 * inv_c2);
            const amrex::ParticleReal inv_gamma = 1.0_prt / gamma;
            amrex::ParticleReal const damping = exp(-nu * inv_gamma * dt);

            ux[ip] *= damping;
            uy[ip] *= damping;
            uz[ip] *= damping;

        }
        );
}
