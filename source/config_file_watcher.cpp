//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "stdafx.h"
#include "config_file_watcher.h"

#include <cassert>

#include <process.h>

namespace
{
void CloseHandleWrapped(void* h)
{
    if (h)
        CloseHandle(h);
}
} // Anonymous namespace

ConfigFileWatcher::ConfigFileWatcher()
    : exit_event_(CreateEvent(nullptr, TRUE, FALSE, nullptr),
                  CloseHandleWrapped)
    , folder_handle_(nullptr, CloseHandleWrapped)
    , thread_handle_(nullptr)
    , lock_(0)
    , file_modified_(false)
{

}

ConfigFileWatcher::~ConfigFileWatcher()
{
    StopWatching();
}

void ConfigFileWatcher::ResetState()
{
    for (;;) {
        if (InterlockedCompareExchange(&lock_, 1, 0) == 0) {
            file_modified_ = 0;
            InterlockedCompareExchange(&lock_, 0, 1);

            return;
        }
        Sleep(0);
    }
}

void ConfigFileWatcher::StartWatching(const std::string& folder)
{
    assert(!thread_handle_);
    if (thread_handle_)
        return;

    folder_handle_.reset(
        CreateFileA(
            folder.c_str(), FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr));

    thread_handle_ = reinterpret_cast<void*>(
        _beginthreadex(nullptr, 0, ThreadProc, this, 0, nullptr));
}

void ConfigFileWatcher::StopWatching()
{
    SetEvent(exit_event_.get());
    if (thread_handle_) {
        WaitForSingleObject(thread_handle_, INFINITE);
        CloseHandle(thread_handle_);
        thread_handle_ = nullptr;
    }
}

unsigned int __stdcall ConfigFileWatcher::ThreadProc(void* param)
{
    ConfigFileWatcher* obj = reinterpret_cast<ConfigFileWatcher*>(param);

    const int buf_size = 1024;
    uint8_t buffer[buf_size];
    std::unique_ptr<void, void (__cdecl*)(void*)> event(
        CreateEvent(nullptr, FALSE, FALSE, nullptr), CloseHandleWrapped);
    if (obj->folder_handle_.get() &&
            obj->folder_handle_.get() != INVALID_HANDLE_VALUE) {
        do {
            DWORD bytes_read = 0;
            OVERLAPPED overlapped = {};
            overlapped.hEvent = event.get();

            BOOL result = ReadDirectoryChangesW(
                obj->folder_handle_.get(), buffer, buf_size, false,
                FILE_NOTIFY_CHANGE_LAST_WRITE, &bytes_read, &overlapped,
                nullptr);

            HANDLE events[] = {obj->exit_event_.get(), overlapped.hEvent};
            DWORD wait_result =
                WaitForMultipleObjects(2, events, FALSE, INFINITE);
            if (wait_result == WAIT_OBJECT_0)
                break;

            DWORD bytes_transferred = 0;
            if (GetOverlappedResult(obj->folder_handle_.get(), &overlapped,
                                    &bytes_transferred, FALSE)) {
                if (bytes_transferred >= sizeof(FILE_NOTIFY_INFORMATION)) {
                    FILE_NOTIFY_INFORMATION* info =
                        reinterpret_cast<FILE_NOTIFY_INFORMATION*>(buffer);
                    if (info->Action == FILE_ACTION_MODIFIED)
                        obj->SetModified();
                }
            }
        } while (true);
    }

    return 0;
}

void ConfigFileWatcher::SetModified()
{
    for (;;) {
        if (InterlockedCompareExchange(&lock_, 1, 0) == 0) {
            file_modified_ = 1;
            InterlockedCompareExchange(&lock_, 0, 1);

            return;
        }
        Sleep(0);
    }
}
