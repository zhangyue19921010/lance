use crate::stats::STATS;
use libc::{c_void, size_t};

extern "C" {
    #[link_name = "__libc_malloc"]
    fn libc_malloc(size: size_t) -> *mut c_void;
    #[link_name = "__libc_calloc"]
    fn libc_calloc(count: size_t, element_size: size_t) -> *mut c_void;
    #[link_name = "__libc_realloc"]
    fn libc_realloc(ptr: *mut c_void, size: size_t) -> *mut c_void;
    #[link_name = "__libc_free"]
    fn libc_free(ptr: *mut c_void);
    #[link_name = "__libc_memalign"]
    fn libc_memalign(alignment: size_t, size: size_t) -> *mut c_void;
}

// Magic number to identify our allocations
const MAGIC: u64 = 0xDEADBEEF_CAFEBABE;

/// Header stored before each tracked allocation
#[repr(C)]
struct AllocationHeader {
    magic: u64,
    size: u64,
    alignment: u64,
    /// For aligned allocations, stores the actual pointer returned by libc_memalign
    /// For unaligned allocations, this is unused (but present for consistent size)
    actual_ptr: u64,
}

const HEADER_SIZE: usize = std::mem::size_of::<AllocationHeader>();

/// Check if a pointer was allocated by us
unsafe fn is_ours(virtual_ptr: *mut c_void) -> bool {
    if virtual_ptr.is_null() {
        return false;
    }
    let header_ptr = (virtual_ptr as *mut u8).sub(HEADER_SIZE) as *const AllocationHeader;
    (*header_ptr).magic == MAGIC
}

/// Extract size, alignment, and actual pointer from a virtual pointer
unsafe fn extract(virtual_ptr: *mut c_void) -> (usize, usize, *mut c_void) {
    let header_ptr = (virtual_ptr as *mut u8).sub(HEADER_SIZE) as *const AllocationHeader;
    let header = &*header_ptr;

    let size = header.size as usize;
    let alignment = header.alignment as usize;

    let actual_ptr = if alignment > 0 {
        // For aligned allocations, the actual pointer is stored in the header
        header.actual_ptr as *mut c_void
    } else {
        // For unaligned allocations, the actual pointer is the header itself
        header_ptr as *mut c_void
    };

    (size, alignment, actual_ptr)
}

/// Take an allocated pointer and size, store header, and return the adjusted pointer
unsafe fn to_virtual(actual_ptr: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
    if actual_ptr.is_null() {
        return std::ptr::null_mut();
    }

    if alignment > 0 {
        // For aligned allocations:
        // 1. Find the first aligned position after we have room for the header
        // 2. Store the header just before that position
        // 3. Store the actual_ptr in the header so we can free it later

        let actual_addr = actual_ptr as usize;
        // Find the first address >= actual_addr + HEADER_SIZE that is aligned
        let min_virtual_addr = actual_addr.saturating_add(HEADER_SIZE);
        let virtual_addr = (min_virtual_addr.saturating_add(alignment).saturating_sub(1))
            & !(alignment.saturating_sub(1));

        // Write header just before the aligned virtual address
        let header_ptr = (virtual_addr.saturating_sub(HEADER_SIZE)) as *mut AllocationHeader;
        *header_ptr = AllocationHeader {
            magic: MAGIC,
            size: size as u64,
            alignment: alignment as u64,
            actual_ptr: actual_addr as u64,
        };

        virtual_addr as *mut c_void
    } else {
        // Unaligned allocation - header is at the start
        let header_ptr = actual_ptr as *mut AllocationHeader;
        *header_ptr = AllocationHeader {
            magic: MAGIC,
            size: size as u64,
            alignment: 0,
            actual_ptr: 0, // Unused for unaligned allocations
        };
        (actual_ptr as *mut u8).add(HEADER_SIZE) as *mut c_void
    }
}

#[no_mangle]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    to_virtual(libc_malloc(size.saturating_add(HEADER_SIZE)), size, 0)
}

#[no_mangle]
pub unsafe extern "C" fn calloc(size: size_t, element_size: size_t) -> *mut c_void {
    let Some(total_size) = size.checked_mul(element_size) else {
        return std::ptr::null_mut();
    };
    STATS.record_allocation(total_size);
    to_virtual(
        libc_calloc(total_size.saturating_add(HEADER_SIZE), 1),
        total_size,
        0,
    )
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    if is_ours(ptr) {
        // It's ours - extract size and track
        let (size, _alignment, actual_ptr) = extract(ptr);
        STATS.record_deallocation(size);
        libc_free(actual_ptr);
    } else {
        // Not ours - just free it without tracking
        libc_free(ptr);
    }
}

#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
    let (old_size, actual_ptr) = if ptr.is_null() || !is_ours(ptr) {
        // Either null or not ours - don't track
        if ptr.is_null() {
            (0, std::ptr::null_mut())
        } else {
            // Not ours - just realloc without tracking
            return libc_realloc(ptr, size);
        }
    } else {
        let (s, _align, a) = extract(ptr);
        (s, a)
    };

    STATS.record_deallocation(old_size);
    STATS.record_allocation(size);

    let new_ptr = libc_realloc(actual_ptr, size.saturating_add(HEADER_SIZE));
    to_virtual(new_ptr, size, 0)
}

#[no_mangle]
pub unsafe extern "C" fn memalign(alignment: size_t, size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    // Allocate extra space for header + padding to maintain alignment
    // We need: header (24 bytes) + actual_ptr (8 bytes) + padding to reach alignment
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = libc_memalign(alignment, size.saturating_add(extra));
    to_virtual(actual_ptr, size, alignment)
}

#[no_mangle]
pub unsafe extern "C" fn posix_memalign(
    memptr: *mut *mut c_void,
    alignment: size_t,
    size: size_t,
) -> i32 {
    STATS.record_allocation(size);
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = libc_memalign(alignment, size.saturating_add(extra));
    if actual_ptr.is_null() {
        return libc::ENOMEM;
    }
    *memptr = to_virtual(actual_ptr, size, alignment);
    0
}

#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = libc_memalign(alignment, size.saturating_add(extra));
    to_virtual(actual_ptr, size, alignment)
}

#[no_mangle]
pub unsafe extern "C" fn valloc(size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    let page_size = libc::sysconf(libc::_SC_PAGESIZE) as size_t;
    let extra = page_size.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = libc_memalign(page_size, size.saturating_add(extra));
    to_virtual(actual_ptr, size, page_size)
}

#[no_mangle]
pub unsafe extern "C" fn reallocarray(
    old_ptr: *mut c_void,
    count: size_t,
    element_size: size_t,
) -> *mut c_void {
    let Some(size) = count.checked_mul(element_size) else {
        return std::ptr::null_mut();
    };
    realloc(old_ptr, size)
}

#[no_mangle]
pub unsafe extern "C" fn malloc_usable_size(ptr: *mut c_void) -> size_t {
    if ptr.is_null() {
        return 0;
    }

    if is_ours(ptr) {
        let (size, _, _) = extract(ptr);
        size
    } else {
        // Not our allocation - return 0 as we don't know the size
        // (there's no __libc_malloc_usable_size to call)
        0
    }
}
