#ifndef _FLUID_CONFIG_H_
#define _FLUID_CONFIG_H_

#include <string>

#include "graphics_lib_enum.h"
#include "fluid_simulator.h"

class FluidConfig
{
public:
    template <typename T>
    struct ConfigField
    {
        T value_;
        const char* desc_;

        ConfigField(T v, const char* d) : value_(v), desc_(d) {}
    };

    static FluidConfig* Instance();

    void CreateIfNeeded(const std::string& path);
    void Load(const std::string& path);
    void Reload();

    GraphicsLib graphics_lib() const { return graphics_lib_.value_; }
    FluidSimulator::PoissonMethod poisson_method() const {
        return poisson_method_.value_;
    }
    float ambient_temperature() const { return ambient_temperature_.value_; }
    float impulse_temperature() const { return impulse_temperature_.value_; }
    float impulse_density() const { return impulse_density_.value_; }
    float smoke_buoyancy() const { return smoke_buoyancy_.value_; }
    float smoke_weight() const { return smoke_weight_.value_; }
    float temperature_dissipation() const {
        return temperature_dissipation_.value_;
    }
    float velocity_dissipation() const { return velocity_dissipation_.value_; }
    float density_dissipation() const { return density_dissipation_.value_; }
    float splat_radius_factor() const { return splat_radius_factor_.value_; }
    int num_jacobi_iterations() const { return num_jacobi_iterations_.value_; }
    int num_full_multigrid_iterations() const {
        return num_full_multigrid_iterations_.value_;
    }

private:
    FluidConfig();
    ~FluidConfig();

    void Parse(const std::string& key, const std::string& value);
    void Store(std::ostream& stream);

    std::string file_path_;
    ConfigField<GraphicsLib> graphics_lib_;
    ConfigField<FluidSimulator::PoissonMethod> poisson_method_;
    ConfigField<float> ambient_temperature_;
    ConfigField<float> impulse_temperature_;
    ConfigField<float> impulse_density_;
    ConfigField<float> smoke_buoyancy_;
    ConfigField<float> smoke_weight_;
    ConfigField<float> temperature_dissipation_;
    ConfigField<float> velocity_dissipation_;
    ConfigField<float> density_dissipation_;
    ConfigField<float> splat_radius_factor_;
    ConfigField<int> num_jacobi_iterations_;
    ConfigField<int> num_full_multigrid_iterations_;
};

#endif // _FLUID_CONFIG_H_