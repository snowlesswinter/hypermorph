#ifndef _MULTIGRID_STAGGERED_SHADER_H_
#define _MULTIGRID_STAGGERED_SHADER_H_

#include <string>

class MultigridStaggeredShader
{
public:
    static std::string RestrictResidualPacked();
    static std::string RestrictPacked();
};

#endif // _MULTIGRID_STAGGERED_SHADER_H_