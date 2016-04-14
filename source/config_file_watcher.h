#ifndef _CONFIG_FILE_WATCHER_H_
#define _CONFIG_FILE_WATCHER_H_

#include <memory>
#include <string>

class ConfigFileWatcher
{
public:
    ConfigFileWatcher();
    ~ConfigFileWatcher();

    void ResetState();
    void StartWatching(const std::string& folder);
    void StopWatching();

    bool file_modified() const { return file_modified_; }

private:
    static unsigned int __stdcall ThreadProc(void* param);

    void SetModified();

    std::unique_ptr<void, void (__cdecl*)(void*)> exit_event_;
    std::unique_ptr<void, void(__cdecl*)(void*)> folder_handle_;
    void* thread_handle_;
    unsigned int lock_;
    bool file_modified_;
};

#endif // _CONFIG_FILE_WATCHER_H_