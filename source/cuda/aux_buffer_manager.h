#ifndef _AUX_BUFFER_MANAGER_H_
#define _AUX_BUFFER_MANAGER_H_

#include <list>
#include <map>
#include <memory>

class AuxBufferManager
{
public:
    AuxBufferManager();
    ~AuxBufferManager();

    void* Allocate(int size);
    void Free(void* p);

private:
    struct DevMemDeleter;

    typedef std::unique_ptr<void, DevMemDeleter> DevMemPtr;

    AuxBufferManager(const AuxBufferManager&);
    void operator=(const AuxBufferManager&);

    std::map<int, std::list<DevMemPtr>> free_;
    std::map<void*, int> in_use_;
};

#endif // _AUX_BUFFER_MANAGER_H_