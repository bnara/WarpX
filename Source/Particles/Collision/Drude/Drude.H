/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_DRUDE_H_
#define WARPX_PARTICLES_COLLISION_DRUDE_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

class Drude final
    : public CollisionBase
{
public:
    Drude (const std::string& collision_name);

    ~Drude () override = default;

    Drude ( Drude const &)             = delete;
    Drude& operator= ( Drude const & ) = delete;
    Drude ( Drude&& )                  = delete;
    Drude& operator= ( Drude&& )       = delete;

    /** Perform the stopping calculation
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform the stopping calculation within a tile for stopping on electrons
     *
     * @param pti particle iterator
     * @param dt time step size
     * @param t current simulation time
     * @param species_mass mass of the active species
     * @param species_charge charge of the active species
     *
     */
    void doDrudeWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                            amrex::ParticleReal species_mass, amrex::ParticleReal species_charge);
private:
};

#endif // WARPX_PARTICLES_COLLISION_DRUDE_H_
