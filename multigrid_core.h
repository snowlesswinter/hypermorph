#ifndef _MULTIGRID_CORE_H_
#define _MULTIGRID_CORE_H_

class GLTexture;
class MultigridCore
{
public:
    MultigridCore();
    ~MultigridCore();

    void Absolute(const GLTexture& texture);

private:
};

#endif // _MULTIGRID_CORE_H_