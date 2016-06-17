#include "stdafx.h"
#include "fluid_config.h"

#include <algorithm>
#include <cctype>
#include <locale>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#include <string>

namespace
{
// trim from start (in place)
static inline void ltrim(std::string &s)
{
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s)
{
    s.erase(
        std::find_if(s.rbegin(), s.rend(),
                     std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
        s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}

// trim from both ends (copying)
static inline std::string trimmed(std::string s)
{
    trim(s);
    return s;
}

static inline std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

struct { GraphicsLib graphics_lib_; char* desc_; } lib_enum_desc[] = {
    {GRAPHICS_LIB_CUDA, "cuda"},
    {GRAPHICS_LIB_GLSL, "glsl"},
    {GRAPHICS_LIB_CUDA_DIAGNOSIS, "diagnosis"},
};

struct { FluidSimulator::PoissonMethod m_; char* desc_; } method_enum_desc[] = {
    {FluidSimulator::POISSON_SOLVER_JACOBI, "j"},
    {FluidSimulator::POISSON_SOLVER_DAMPED_JACOBI, "dj"},
    {FluidSimulator::POISSON_SOLVER_MULTI_GRID, "mg"},
    {FluidSimulator::POISSON_SOLVER_FULL_MULTI_GRID, "fmg"},
    {FluidSimulator::POISSON_SOLVER_MULTI_GRID_PRECONDITIONED_CONJUGATE_GRADIENT, "mgpcg"},
};

struct { CudaMain::AdvectionMethod m_; char* desc_; } advect_enum_desc[] = {
    {CudaMain::SEMI_LAGRANGIAN, "sl"},
    {CudaMain::MACCORMACK_SEMI_LAGRANGIAN, "mcsl"},
    {CudaMain::BFECC_SEMI_LAGRANGIAN, "bfeccsl"},
};

template <typename T>
std::istream& operator >>(std::istream& is, FluidConfig::ConfigField<T>& field)
{
    return is >> field.value_;
}

template <>
std::istream& operator >> <GraphicsLib>(
    std::istream& is, FluidConfig::ConfigField<GraphicsLib>& field)
{
    std::string lib;
    std::getline(is, lib);
    std::string lower_trimmed = to_lower(trimmed(lib));
    for (auto i : lib_enum_desc)
        if (lower_trimmed == i.desc_)
            field.value_ = i.graphics_lib_;

    return is;
}

template <>
std::istream& operator >> <FluidSimulator::PoissonMethod>(
    std::istream& is,
    FluidConfig::ConfigField<FluidSimulator::PoissonMethod>& field)
{
    std::string method;
    std::getline(is, method);
    std::string lower_trimmed = to_lower(trimmed(method));
    for (auto i : method_enum_desc)
        if (lower_trimmed == i.desc_)
            field.value_ = i.m_;

    return is;
}

template <>
std::istream& operator >> <CudaMain::AdvectionMethod>(
    std::istream& is,
    FluidConfig::ConfigField<CudaMain::AdvectionMethod>& field)
{
    std::string method;
    std::getline(is, method);
    std::string lower_trimmed = to_lower(trimmed(method));
    for (auto i : advect_enum_desc)
        if (lower_trimmed == i.desc_)
            field.value_ = i.m_;

    return is;
}

template <>
std::istream& operator >> <glm::vec3>(
    std::istream& is, FluidConfig::ConfigField<glm::vec3>& field)
{
    std::string color;
    std::getline(is, color);
    std::string trimmed_color = trimmed(color);
    std::regex re(
        "\\(\\s*([0-9]{1,3})\\s*,\\s*([0-9]{1,3})\\s*,\\s*([0-9]{1,3})\\s*\\)");
    std::smatch matched;
    if (std::regex_match(trimmed_color, matched, re)) {
        if (matched.size() == 4) {
            for (size_t i = 1; i < matched.size(); i++) {
                std::stringstream ss(matched[i]);
                ss >> field.value_[i - 1];
            }
        }
    }

    return is;
}

template <typename T>
std::ostream& operator <<(std::ostream& os, FluidConfig::ConfigField<T>& field)
{
    return os << field.desc_ << " = " << field.value_;
}

template <>
std::ostream& operator << <GraphicsLib>(
    std::ostream& os, FluidConfig::ConfigField<GraphicsLib>& field)
{
    os << field.desc_ << " = ";
    for (auto i : lib_enum_desc)
        if (field.value_ == i.graphics_lib_)
            os << i.desc_;

    return os;
}

template <>
std::ostream& operator << <FluidSimulator::PoissonMethod>(
    std::ostream& os,
    FluidConfig::ConfigField<FluidSimulator::PoissonMethod>& field)
{
    os << field.desc_ << " = ";
    for (auto i : method_enum_desc)
        if (field.value_ == i.m_)
            os << i.desc_;

    return os;
}

template <>
std::ostream& operator << <CudaMain::AdvectionMethod>(
    std::ostream& os,
    FluidConfig::ConfigField<CudaMain::AdvectionMethod>& field)
{
    os << field.desc_ << " = ";
    for (auto i : advect_enum_desc)
        if (field.value_ == i.m_)
            os << i.desc_;

    return os;
}

template <>
std::ostream& operator << <glm::vec3>(
    std::ostream& os, FluidConfig::ConfigField<glm::vec3>& field)
{
    os << field.desc_ << " = (" << field.value_.x << ", " << field.value_.y <<
        ", " << field.value_.z << ")";

    return os;
}

} // Anonymous namespace.

FluidConfig* FluidConfig::Instance()
{
    static FluidConfig* m = nullptr;
    if (!m)
        m = new FluidConfig();

    return m;
}

void FluidConfig::CreateIfNeeded(const std::string& path)
{
    if (std::ifstream(path))
        return;

    std::ofstream file_stream(path);
    if (file_stream) {
        Store(file_stream);
    }
}

void FluidConfig::Load(const std::string& path, const std::string& preset_path)
{
    file_path_ = path;
    preset_path_ = preset_path;

    preset_file_.value_.clear();
    Load(file_path_);

    if (!preset_file_.value_.empty())
        Load(preset_path_ + "\\" + preset_file_.value_);
}

void FluidConfig::Reload()
{
    if (file_path_.empty())
        return;

    Load(file_path_, preset_path_);
}

FluidConfig::FluidConfig()
    : file_path_()
    , preset_path_()
    , preset_file_("", "preset")
    , graphics_lib_(GRAPHICS_LIB_CUDA, "graphics library")
    , poisson_method_(FluidSimulator::POISSON_SOLVER_FULL_MULTI_GRID,
                      "poisson method")
    , advection_method_(CudaMain::MACCORMACK_SEMI_LAGRANGIAN,
                        "advection method")
    , light_color_(glm::vec3(171, 160, 139), "light color")
    , grid_size_(glm::vec3(128, 128, 128), "grid size")
    , cell_size_(0.15f, "cell size")
    , ambient_temperature_(0.0f, "ambient temperature")
    , impulse_temperature_(40.0f, "impulse temperature")
    , impulse_density_(0.5f, "impulse density")
    , impulse_velocity_(10.0f, "impulse velocity")
    , smoke_buoyancy_(1.0f, "smoke buoyancy")
    , smoke_weight_(0.0001f, "smoke weight")
    , temperature_dissipation_(0.15f, "temperature dissipation")
    , velocity_dissipation_(0.001f, "velocity dissipation")
    , density_dissipation_(0.2f, "density dissipation")
    , splat_radius_factor_(0.25f, "splat radius factor")
    , fixed_time_step_(0.33f, "fixed time step")
    , light_intensity_(22.0f, "light intensity")
    , light_absorption_(10.0f, "light absorption")
    , raycast_density_factor_(30.0f, "raycast density factor")
    , raycast_occlusion_factor_(15.0f, "raycast occlusion factor")
    , time_stretch_(1.0f, "time stretch")
    , vorticity_confinement_(0.1f, "vorticity confinement")
    , num_jacobi_iterations_(40, "number of jacobi iterations")
    , num_multigrid_iterations_(5, "num multigrid iterations")
    , num_full_multigrid_iterations_(2, "num full multigrid iterations")
    , num_mgpcg_iterations_(2, "num mgpcg iterations")
    , auto_impulse_(1, "auto impulse")
    , staggered_(1, "staggered")
    , mid_point_(0, "mid point")
    , num_raycast_samples_(224, "num raycast samples")
    , num_raycast_light_samples_(64, "num raycast light samples")
    , initial_viewport_width_(512)
    , initial_viewport_height_(512)
{
}

FluidConfig::~FluidConfig()
{
}

void FluidConfig::Load(const std::string& file_path)
{
    std::ifstream file_stream(file_path);
    std::string line;
    while (std::getline(file_stream, line))
    {
        std::stringstream line_stream(line);
        std::string key;
        if (std::getline(line_stream, key, '='))
        {
            std::string value;
            std::getline(line_stream, value);
            Parse(key, value);
        }
    }
}

void FluidConfig::Parse(const std::string& key, const std::string& value)
{
    std::string lower_trimmed = to_lower(trimmed(key));
    std::stringstream value_stream(value);
    if (lower_trimmed == graphics_lib_.desc_) {
        value_stream >> graphics_lib_;
        return;
    }
    
    if (lower_trimmed == poisson_method_.desc_) {
        value_stream >> poisson_method_;
        return;
    }

    if (lower_trimmed == advection_method_.desc_) {
        value_stream >> advection_method_;
        return;
    }

    if (lower_trimmed == light_color_.desc_) {
        value_stream >> light_color_;
        return;
    }

    if (lower_trimmed == grid_size_.desc_) {
        value_stream >> grid_size_;
        return;
    }

    ConfigField<std::string>* string_fields[] = {
        &preset_file_,
    };

    for (auto& f : string_fields) {
        if (lower_trimmed == f->desc_) {
            value_stream >> *f;
            return;
        }
    }

    ConfigField<float>* float_fields[] = {
        &cell_size_,
        &ambient_temperature_,
        &impulse_temperature_,
        &impulse_density_,
        &impulse_velocity_,
        &smoke_buoyancy_,
        &smoke_weight_,
        &temperature_dissipation_,
        &velocity_dissipation_,
        &density_dissipation_,
        &splat_radius_factor_,
        &fixed_time_step_,
        &light_intensity_,
        &light_absorption_,
        &raycast_density_factor_,
        &raycast_occlusion_factor_,
        &time_stretch_,
        &vorticity_confinement_,
    };

    for (auto& f : float_fields) {
        if (lower_trimmed == f->desc_) {
            value_stream >> *f;
            return;
        }
    }

    ConfigField<int>* int_fields[] = {
        &num_jacobi_iterations_,
        &num_multigrid_iterations_,
        &num_full_multigrid_iterations_,
        &num_mgpcg_iterations_,
        &auto_impulse_,
        &staggered_,
        &mid_point_,
        &num_raycast_samples_,
        &num_raycast_light_samples_,
    };

    for (auto& f : int_fields) {
        if (lower_trimmed == f->desc_) {
            value_stream >> *f;
            return;
        }
    }
}

void FluidConfig::Store(std::ostream& stream)
{
    stream << graphics_lib_ << std::endl;
    stream << poisson_method_ << std::endl;
    stream << advection_method_ << std::endl;
    stream << light_color_ << std::endl;
    stream << grid_size_ << std::endl;

    ConfigField<std::string> string_fields[] = {
        preset_file_,
    };

    for (auto& f : string_fields)
        stream << f << std::endl;

    ConfigField<float> float_fields[] = {
        cell_size_,
        ambient_temperature_,
        impulse_temperature_,
        impulse_density_,
        impulse_velocity_,
        smoke_buoyancy_,
        smoke_weight_,
        temperature_dissipation_,
        velocity_dissipation_,
        density_dissipation_,
        splat_radius_factor_,
        fixed_time_step_,
        light_intensity_,
        light_absorption_,
        raycast_density_factor_,
        raycast_occlusion_factor_,
        time_stretch_,
        vorticity_confinement_,
    };

    for (auto& f : float_fields)
        stream << f << std::endl;

    ConfigField<int> int_fields[] = {
        num_jacobi_iterations_,
        num_multigrid_iterations_,
        num_full_multigrid_iterations_,
        num_mgpcg_iterations_,
        auto_impulse_,
        staggered_,
        mid_point_,
        num_raycast_samples_,
        num_raycast_light_samples_,
    };

    for (auto& f : int_fields)
        stream << f << std::endl;
}
