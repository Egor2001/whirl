#pragma once

#include <assert.h>
#include <memory>
#include <thread>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "Windows.h"

#include "WHBuffer.h"

#define __WH_WAIT_FOR(cond) while (!cond) Sleep(0);

class WHWindow;

class WHWindow
{
private:
    class ref_counter;

public:
    WHWindow() = default;
    WHWindow(const SIZE& size_set);

    virtual ~WHWindow();

public:
    HWND create (const SIZE& size_set);
    void destroy();

    template<WHMemoryLocation MEMORY_LOCATION> int flush	 (const WHBuffer<MEMORY_LOCATION>* buffer);
	template<WHMemoryLocation MEMORY_LOCATION> int flush_back(      WHBuffer<MEMORY_LOCATION>* buffer);
        
    HWND get_handle  () const { return wnd_handle_; }
    SIZE get_wnd_size() const { return wnd_size_; }

    void show() const { UpdateWindow(wnd_handle_); ShowWindow(wnd_handle_, SW_SHOW); }
    void hide() const { UpdateWindow(wnd_handle_); ShowWindow(wnd_handle_, SW_HIDE); }

    template<WHMemoryLocation MEMORY_LOCATION>
    WHBuffer<MEMORY_LOCATION> create_buffer() { return WHBuffer<MEMORY_LOCATION>(wnd_size_); }

private:
    void thread_proc_();
    
    static LRESULT CALLBACK wnd_proc_(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam);
    
    LRESULT msg_create_proc_ (HWND wnd, WPARAM wparam, LPARAM lparam);
    LRESULT msg_destroy_proc_(HWND wnd, WPARAM wparam, LPARAM lparam);
    LRESULT msg_close_proc_  (HWND wnd, WPARAM wparam, LPARAM lparam);
    LRESULT msg_paint_proc_  (HWND wnd, WPARAM wparam, LPARAM lparam);
    
private:
    static const LPCTSTR    WND_CLASS_NAME_;
    static const WNDCLASSEX WND_CLASS_EX_;
    static const ATOM       WND_CLASS_ATOM_;

    std::mutex  mutex_;
	std::thread msg_thread_;
	HWND        wnd_handle_;
	SIZE        wnd_size_;
    HDC         mirror_dc_;
};

class WHWindow::ref_counter
{
public:
    static void add_ref(WHWindow* ref)
    {
        refs.push_back(ref);
    }

    static void remove_ref(const WHWindow* ref)
    {
        auto iter = std::find(refs.begin(), refs.end(), ref);

        if (iter != refs.end()) refs.erase(iter);
    }

    static WHWindow* getWindowByHandle(HWND wnd_h)
    {
        auto iter = std::find_if(refs.begin(), refs.end(), [wnd_h](WHWindow* wnd_ptr) { return wnd_ptr->get_handle() == wnd_h; });

        if (iter == refs.end()) return nullptr;

        return *iter;
    }

private:
    ref_counter () = delete;
    ~ref_counter() = delete;

    static std::vector<WHWindow*> refs;
};

std::vector<WHWindow*> WHWindow::ref_counter::refs;

const LPCTSTR WHWindow::WND_CLASS_NAME_ = "ELL_WINDOW_CLASS";

const WNDCLASSEX WHWindow::WND_CLASS_EX_ = { sizeof(WNDCLASSEX), (CS_OWNDC | CS_HREDRAW | CS_VREDRAW), &WHWindow::wnd_proc_, 
                                             0, 0, nullptr, nullptr, LoadCursor(nullptr, IDC_HAND), 
                                             static_cast<HBRUSH>(GetStockObject(HOLLOW_BRUSH)), 
                                             nullptr, WHWindow::WND_CLASS_NAME_, nullptr };

const ATOM WHWindow::WND_CLASS_ATOM_ = RegisterClassEx(&WHWindow::WND_CLASS_EX_);

WHWindow::WHWindow(const SIZE& size_set): WHWindow()
{
    create(size_set);
}

WHWindow::~WHWindow()
{
    destroy();
}

HWND WHWindow::create(const SIZE& size_set)
{
    wnd_size_ = size_set;
    
    WHWindow::ref_counter::add_ref(this);
    
    msg_thread_ = std::thread(&WHWindow::thread_proc_, this);
    
    __WH_WAIT_FOR(wnd_handle_);

    return wnd_handle_;
}

void WHWindow::destroy()
{
    if (wnd_handle_) 
        DestroyWindow(wnd_handle_);
    
    if (msg_thread_.joinable())
        msg_thread_.join();
    
    wnd_handle_ = nullptr;

    WHWindow::ref_counter::remove_ref(this);
}

void WHWindow::thread_proc_()
{   
    wnd_handle_ = CreateWindowEx(WS_EX_APPWINDOW | WS_EX_CLIENTEDGE, WND_CLASS_NAME_, nullptr, 
                                 (WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_BORDER)),
                                 (GetSystemMetrics(SM_CXSCREEN) - wnd_size_.cx)/2, (GetSystemMetrics(SM_CYSCREEN) - wnd_size_.cy)/2, 
                                 wnd_size_.cx, wnd_size_.cy, nullptr, nullptr, nullptr, static_cast<LPVOID>(this)); assert(wnd_handle_);

    UpdateWindow(wnd_handle_);
    ShowWindow  (wnd_handle_, SW_SHOW);

    MSG message = {};

    while (GetMessage(&message, nullptr, 0, 0))
    {
        TranslateMessage(&message);
        DispatchMessage (&message);
        
        Sleep(0);
    }
}

LRESULT CALLBACK WHWindow::wnd_proc_(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    WHWindow* wnd_ptr = nullptr;

    if (msg == WM_CREATE) wnd_ptr = static_cast<WHWindow*>(reinterpret_cast<LPCREATESTRUCT>(lparam)->lpCreateParams); 
    else                  wnd_ptr = WHWindow::ref_counter::getWindowByHandle(wnd);

	switch (msg)
	{
    case WM_CREATE:  return wnd_ptr->msg_create_proc_ (wnd, wparam, lparam);
    case WM_DESTROY: return wnd_ptr->msg_destroy_proc_(wnd, wparam, lparam);
	case WM_CLOSE:   return wnd_ptr->msg_close_proc_  (wnd, wparam, lparam);
	case WM_PAINT:   return wnd_ptr->msg_paint_proc_  (wnd, wparam, lparam);

	default: return DefWindowProc(wnd, msg, wparam, lparam);
	}
}

LRESULT WHWindow::msg_create_proc_(HWND wnd, WPARAM wparam, LPARAM lparam)
{
    RECT paintArea   = {};
    SIZE paintAreaSz = {};

	GetClientRect(wnd, &paintArea);

	paintAreaSz = { paintArea.right  - paintArea.left,
				    paintArea.bottom - paintArea.top };

	mutex_.lock();
	{
		HDC     wnd_dc = GetDC(wnd);                                                     assert(wnd_dc);
		HBITMAP bmp    = CreateCompatibleBitmap(wnd_dc, paintAreaSz.cx, paintAreaSz.cy); assert(bmp);
		
		mirror_dc_ = CreateCompatibleDC(wnd_dc); assert(mirror_dc_);

		SelectObject(mirror_dc_, bmp);
	}
	mutex_.unlock();

	return 0;
}

LRESULT WHWindow::msg_destroy_proc_(HWND wnd, WPARAM wparam, LPARAM lparam)
{
	mutex_.lock();
    {
		DeleteDC(mirror_dc_);
    }
    mutex_.unlock();

	PostQuitMessage(0);

    return 0;
}

LRESULT WHWindow::msg_close_proc_(HWND wnd, WPARAM wparam, LPARAM lparam)
{
	SendNotifyMessage(wnd, WM_DESTROY, 0, 0);

	return 0;
}

LRESULT WHWindow::msg_paint_proc_(HWND wnd, WPARAM wparam, LPARAM lparam)
{
    PAINTSTRUCT paintStruct = {};
    RECT		paintArea   = {};
    SIZE        paintAreaSz = {};

	HDC draw_dc = BeginPaint(wnd, &paintStruct); assert(draw_dc);
	GetClientRect(wnd, &paintArea);

	paintAreaSz = { paintArea.right  - paintArea.left, 
                    paintArea.bottom - paintArea.top };

	mutex_.lock();
	{
		BitBlt(draw_dc,	0, 0, paintAreaSz.cx, paintAreaSz.cy, mirror_dc_, 0, 0, SRCCOPY);
    }
	mutex_.unlock();

	return !EndPaint(wnd, &paintStruct);
}

template<WHMemoryLocation MEMORY_LOCATION>
int WHWindow::flush(const WHBuffer<MEMORY_LOCATION>* buffer)
{
    int result = 0;

	mutex_.lock();
	{
        result = buffer->set_bytes_to_dc(mirror_dc_);
	}
	mutex_.unlock();
    
    RedrawWindow(wnd_handle_, nullptr, nullptr, RDW_INTERNALPAINT | RDW_INVALIDATE | RDW_UPDATENOW);

    return result;
}

template<WHMemoryLocation MEMORY_LOCATION>
int WHWindow::flush_back(WHBuffer<MEMORY_LOCATION>* buffer)
{
    int result = 0;

	mutex_.lock();
	{
        result = buffer->get_bytes_from_dc(mirror_dc_);
    }
	mutex_.unlock();

    return result;
}
