

# unique_lock and lock_guard

`std::mutex` is neither copable nor assignable
and `lock_guard` is just a RAII wrapper around `std::mutex`.
In the constructor, it call `lock` and in the destructor it calls
`unlock`.

`unqiue_lock` is like `unique_ptr`, which is movable. It also
has `release`, `unlock` methods. Use this one if we want to lock
multiple mutexes at the same time, i.e., atomically.
