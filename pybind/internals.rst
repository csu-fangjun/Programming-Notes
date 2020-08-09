
Internals
=========

First Commit
------------

Some macros (`include/pybind/common.h`):

.. code-block::

  #define NAMESPACE_BEGIN(name) namespace name {
  #define NAMESPACE_END(name) }

`include/pybind/pytypes.h`:

**handle**
  A ``handle`` wraps a ``PyObject *m_ptr``. ``sizeof(Handle) == sizeof(PyObject *mptr)``.

  The main function of ``handle``:
  1. wrap ``PyObject*``, but does not support reference counting.
  2. the ``bool operator``
  3. ``inc_ref``, ``dec_ref``, ``ref_count``
  4. ``operator []`` with ``detail::accessor`` wrapping ``PyObject_SetItem``, ``PyObject_GetItem``
  4. ``attr()`` with ``detail::accessor`` wrapping ``PyObject_SetAttr``, ``PyObject_GetAttr``

**accessor**
  1. wraps ``PyObject_SetItem`, ``PyObject_GetItem``, ``PyObject_SetAttr``, ``PyObject_GetAttr``.
  2. assign a ``handle`` to ``accessor``
  3. convert a ``accessor`` to ``object``
  4. assign a ``accessor`` to another ``accessor``
  5. ``operator bool``
  6. It needs an ``PyObject* obj`` and a key: ``const char*`` or ``PyObject*``.

**list_accessor**
  1. It is not a subclass of ``accessor``
  2. It needs a ``PyObject* list`` and an index ``size_t index``.
  3. It wraps ``PyList_SetItem`` and ``PyList_GetItem``

**tuple_accessor**
  1. It is not a subclass of ``accessor``
  2. It needs ``PyObject* tuple`` and ``size_t index``.
  3. It wraps ``PyTuple_SetItem`` and ``PyTuple_SetItem``

**tuple**
  1. Uses ``PyTuple_Check``
  2. The tuple must be initialized to a given size in the constructor
  3. ``operator[]()`` returns a ``tuple_accesor``

**dict_iterator**
  1. Uses ``PyObject* dict`` and ``ssize_t pos``. The default value of ``pos`` is -1
  2. It defines the prefix increment operator ``dict_iterator& operator++()``,
       which uses ``PyDict_Next``.
  3. It defines ``std::pair<object, object> operator*()``
  4. For ``operator ==`` and ``operator !=``, it compares only ``pos``

**dict**
  1. ``PyDict_Check``
  2. Uses ``PyDict_New`` for the default constructor.
  3. ``begin()``, returns ``++dict_iterator(ptr(), 0)``. It uses prefix increment
       because we need to initialize ``key`` and ``value`` inside the prefix increment
       operator of ``dict_iterator``.
  4. We can use only `range-for` to access ``dict``
  5. It has a ``size()`` method

.. code-block::

  for(auto item: dict) {
    // note that both `first` and `second` are of type `object`
    std::cout << item.first << " " << item.second << "\n";
  }

**list_iterator**
  1. Uses ``PyObject* list`` and ``ssize_t pos``.
  2. It defines ``list_iterator&  operator++()``.
  3. It defines ``operator*  operator*()``, which calls ``PyList_GetItem``
  4. For ``operator ==`` and ``operator !=``, it compares only ``pos``

**list**
  1. ``PyList_Check``
  2. ``PyList_New`` with a given size (may be 0, e.g., an empty list)
  3. a ``size()`` method, calling ``PyList_Size``
  4. ``begin()`` and ``end()`` for ``range-for`` loop
  5. ``operator[]`` returns a ``list_iterator``
  6. ``append`` method, calling ``PyList_Append``

**str**
  1. ``class str: public class object``
  2. It can be constructed froma and converted to ``const char*`` using
       ``PyUnicode_FromString`` and ``PyUnicode_AsUTF8``
  3. An ``object`` can be converted to a ``str`` using ``PyObject_Str()``
  4. We can use ``std::cout << object`` to print an ``object``, which
       first invokes ``object.str()`` and then calls ``const char*`` convertor.

**capsule**
  1. It wraps a ``void* ptr`` using ``PyCapsule_New``
  2. The pointer can be later retrieved using ``PyCapsule_GetPointer``
  3. It has a template operator conversion ``operator T*()``



internals
---------

``include/pybind/common.h``

.. code-block::

  struct internals {
      std::unordered_map<std::string, type_info> registered_types;
      std::unordered_map<void *, PyObject *> registered_instances;
  };

``get_internals()`` in ``pytypes.h`` will
- 1. get ``PyEval_GetBuiltins``
- 2. if the returned object has an attribute ``__pybind__`` and it is a capsule, then
return the pointer in the capsule
- 3. there is no ``_pybind__`` attribute, use ``new struct internals`` to create a new instance
and wrapped it into a capsule and save it as an attribute of the returned butilins object.

mpl.h
-----

**normalize_type**
  1. T to T
  2. const T to T
  3. T* to T
  4. T& to T
  5. T&& to T
  6. const T[N] to T
  7. T[N] to T

**remove_class**
  1. ``R (C::*)(A...)`` to ``R(A...)``
  2. ``R (C::*)(A...) const`` to ``R(A...)``

**lambda_signature_impl**
  1. For a lambda function of type T, or for a callable object of class T:

    - std::remove_reference<T>::type
    - decltype(&type::operator()), which is a function pointer type to a class member function
    - use ``remove_class`` to get ``R(A...)``

  2. ``R(&)(A...)`` to ``R(A...)``
  3. ``R(*)(A...)`` to ``R(A...)``

  .. code-block::

      template <typename T> using lambda_signature = typename lambda_signature_impl<T>::type;
      template <typename F> using make_function_type = std::function<lambda_signature<F>>;

  - ``lambda_signature<F>`` is ``R(A...)``
  - ``make_function_type<F>`` is ``std::function<R(A...)>``

  .. code-block::

**make_function**
  For any callable, return an instance of ``std::function<R(...)>``.

  .. code-block::

      template<typename F> detail::make_function_type<F> make_function(F &&f) {
          return detail::make_function_type<F>(std::forward<F>(f)); }


**tuple_dispatch**
  The purpose of ``void_type`` is to return ``None``.

  Note that in ``Arg&& args``, ``Arg`` is a tuple.

  .. code-block::

    struct void_type { };

    /// Helper functions for calling a function using a tuple argument while dealing with void/non-void return values
    template <typename RetType> struct tuple_dispatch {
        typedef RetType return_type;
        template<typename Func, typename Arg, size_t ... S> return_type operator()(const Func &f, Arg && args, index_sequence<S...>) {
            return f(std::get<S>(std::forward<Arg>(args))...);
        }
    };

    /// Helper functions for calling a function using a tuple argument (special case for void return values)
    template <> struct tuple_dispatch<void> {
        typedef void_type return_type;
        template<typename Func, typename Arg, size_t ... S> return_type operator()(const Func &f, Arg &&args, index_sequence<S...>) {
            f(std::get<S>(std::forward<Arg>(args))...);
            return return_type();
        }
    };

**function_traits**
  **normal functions**
    1. ``ReturnType(*)(Args...)``, for normal functions
    2. Define an enum: ``nargs=sizeof...(Args)``, ``is_method=0``, ``is_const=0``
    3. Some type alias

         .. code-block::

              typedef std::function<ReturnType (Args...)>    f_type;
              typedef detail::tuple_dispatch<ReturnType>     dispatch_type;
              typedef typename dispatch_type::return_type    return_type;
              typedef std::tuple<Args...>                    args_type;

    4. a struct for getting the type of the ``i-th`` type

        .. code-block::

            template <size_t i> struct arg {
                typedef typename std::tuple_element<i, args_type>::type type;
            };

    5. ``cast``, convert ``R(Args...)`` to ``std::function<R(Args...)>``, e.g., the ``f_type``.

        .. code-block::

            static f_type cast(ReturnType (*func)(Args ...)) { return func; }

        Note that it is a static method.

    6. ``dispatch``, which is a static method

        .. code-block::

            static return_type dispatch(const f_type &f, args_type &&args) {
                return dispatch_type()(f, std::move(args),
                    typename make_index_sequence<nargs>::type());
            }


  **class methods (non-const)**
    1. ``ReturnType(ClassType::*)(Args...)``
    2. ``nargs=sizeof...(Args)``, note that ``this`` is not included!
    3. ``is_method=1``, ``is_const=0``
    4. Some type aliases

         .. code-block::

            typedef std::function<ReturnType(ClassType&, Args...)>  f_type;
            typdef detail::tuple_dispatch<ReturnType>               dispatch_type;
            typename dispatch_type::return_type                     return_type;
            typedef std::tuple<ClassType&, Args...>                 args_type;

    5. a struct

        .. code-block::

          template<size_t i> struct arg{
              typedef typename std::tuple_element<i, args_type>::type type;
          };

    6. a ``cast`` function using ``std::mem_fn``

        .. code-block::

          static f_type cast(ReturnType (ClassType::*func)(Args ...)) { return std::mem_fn(func); }

    7. ``dispatch``

        .. code-block::

          static return_type dispatch(const f_type &f, args_type &&args) {
              return dispatch_type()(f, std::move(args),
                  typename make_index_sequence<nargs+1>::type());
          }

pybind.h
--------

**m.def**

  .. code-block::

      template <typename Func> module& def(const char *name, Func f, const char *doc = nullptr) {
          function func(name, f, false, (function) attr(name), doc);
          func.inc_ref(); /* The following line steals a reference to 'func' */
          PyModule_AddObject(ptr(), name, func.ptr());
          return *this;
      }

**function**
  1. It is a subclass of ``object``. The check function is ``PyFunction_Check``.

  .. code-block::

      template <typename Func>
      function(const char *name, Func _func, bool is_method,
               function overload_sibling = function(), const char *doc = nullptr,
               return_value_policy policy = return_value_policy::automatic) {

  ``(function) attr(name)`` is for overload

  2. Choose a ``function_traits`` depending on ``Func``, which has
  static methods ``cast`` and ``dispatch``. The ``cast`` method convert ``Func _func``
  to a ``std::function`` object.

      .. code-block::

          typedef mpl::function_traits<Func> f_traits;

  3. The type caster for cast input arguments:

        .. code-block::

            typedef typename detail::type_caster<typename f_traits::args_type> cast_in;

     Note that ``f_traits::args_type`` is a ``std::tuple``.


     **include/pybind/cast.h**
        1. ``std::tuple``

            .. code-block::

                template <typename ... Tuple> class type_caster<std::tuple<Tuple...>> {
                    typedef std::tuple<Tuple...> type;

            .. code-block::

                std::tuple<type_caster<typename mpl::normalize_type<Tuple>::type>...> value;

            Note that it uses ``normalize_type`` !

        2. ``load``

            .. code-block::

                bool load(PyObject *src, bool convert) {
                    return load(src, convert, typename mpl::make_index_sequence<sizeof...(Tuple)>::type());
                }

            Note that ``convert`` is ``true``.

            .. code-block::

                template <size_t ... Indices> bool load(PyObject *src, bool convert, mpl::index_sequence<Indices...>) {
                    if (!PyTuple_Check(src))
                        return false;
                    if (PyTuple_Size(src) != size)
                        return false;
                    std::array<bool, size> results {{
                        std::get<Indices>(value).load(PyTuple_GetItem(src, Indices), convert)...
                    }};
                    for (bool r : results)
                        if (!r)
                            return false;
                    return true;
                }

        3. For ``int``, ``int*``, ``const int*``, ``int&``, etc, it uses ``type_caster<int>``.

            .. code-block::

              template<>
              class type_caster<int32_t> {
                public:
                  bool load(PyObject* src, boo) {
                    value = (int32_t) PyLong_AsLong(src);
                    if (value == (int32_t)-1 && PyErr_Occurred()) {
                      PyErr_Clear();
                      return false;
                    }
                    return true;
                  }
                  static PyObject* cast(int32_t src, return_value_policy /*policy*/, PyObject* /*parent*/) {
                    return PyLong_FromLong((long)src);
                  }

                protected:
                  int32_t value;
                public:
                  static std::string name() {return "int32_t";}
                  static PyObject* cast(const int32_t* src, return_value_policy policy, PyObject* parent) {
                    return cast(*src, policy, parent);
                  }
                  operator int32_t*() {return &value;}
                  operator int32_t&() {return value;}
              };

  4. The wrapper function

      .. code-block::

        typedef mpl::function_traits<Func> f_traits;
        typedef typename detail::type_caster<typename f_traits::args_type> cast_in;
        typedef typename detail::type_caster<typename mpl::normalize_type<typename f_traits::return_type>::type> cast_out;

        typename f_traits::f_type func = f_traits::cast(_func);

        auto impl = [func, policy](PyObject *pyArgs) -> PyObject *{
            cast_in args;
            if (!args.load(pyArgs, true))
                return nullptr;
            PyObject *parent = policy != return_value_policy::reference_internal
                ? nullptr : PyTuple_GetItem(pyArgs, 0);
            return cast_out::cast(
                f_traits::dispatch(func, (typename f_traits::args_type) args),
                policy, parent);
        };

     ``func`` is of type ``std::function``, which is captured by the lambda function.

Custom Types
------------

.. code-block::

    template <typename type, typename holder_type = std::unique_ptr<type>> class class_ : public detail::custom_type {
        typedef detail::instance<type, holder_type> instance_type;
        ...
        class_(object &scope, const char *name, const char *doc = nullptr)
            : detail::custom_type(scope, name, type_id<type>(), sizeof(type),
                                  sizeof(instance_type), init_holder, dealloc,
                                  nullptr, doc) { }

.. code-block::

    class custom_type : public object {
        PYTHON_OBJECT_DEFAULT(custom_type, object, PyType_Check)
        custom_type(object &scope, const char *name_, const std::string &type_name,
                    size_t type_size, size_t instance_size,
                    void (*init_holder)(PyObject *), const destructor &dealloc,
                    PyObject *parent, const char *doc) {

- ``tp_basicsize``: ``instance_size``
- ``tp_init``: ``init``
- ``tp_new``: ``new_instance``
- ``tp_dealloc``: ``dealloc``
- ``tp_base``: ``parent``

instance
--------

Inside the ``new_instance``, it uses ``PyType_GenericAlloc`` to allocate space for ``instance``.
``::operator new`` is used to allocate space for ``instance->value``.

.. code-block::

    template <typename type, typename holder_type = std::unique_ptr<type>> struct instance {
        PyObject_HEAD
        type *value;
        PyObject *parent;
        bool owned : 1;
        bool constructed : 1;
        holder_type holder;
    };


.. code-block::

    struct internals {
        std::unordered_map<std::string, type_info> registered_types;
        std::unordered_map<void *, PyObject *> registered_instances;
    };

``instance->value`` is used as the key for ``registered_instances``, the value is ``instance`` itself.

The default ``init`` method of ``Custom`` throws an error, so we have to define ``py::init`` for
a given class by ourselves; or we can define ``__init__`` by ourselves. It will use ``type_caster``
to convert ``instance`` to ``instance->value``.

``py::init<Args...>()`` will use placement new to initialize the object. ``dealloc`` will
destruct the instance.


FAQ
---

1. What does Pybind see for the following code:

   .. code-block::

    int* add(int *a);

   - For the argument, it creates a ``type_caster<int>`` which contains a data member ``int value``.
     It uses ``PyLong_AsLong`` to parse``PyObject* src`` and assign the result
     to ``value``. It uses the ``operator int*`` to return a pointer to ``value`` and uses this
     pointer as the function argument ``int *a``. So everything you do inside the function with the
     argument ``int *a`` is visible only to the ``type_caster<int>`` and it has no effect on the
     original Python object.

   - For the return value, it does not return a ``int*`` pointer to Python; instead, it uses
     ``PyLong_FromLong`` to construct a new ``PyObject*`` from ``int*`` and return this ``PyObject*``.

2. What about the following code

   .. code-block::

    const char* add(const char *a);

   - For the argument, it creates a ``type_caster<char>`` which contains a data member ``char* value``.
     It assumes that the passed python object can be used for ``PyUnicode_AsUTF8()``. It contains
     two conversion operators: ``operator char*()`` and ``operator char()``.

   - For the return value, it uses ``PyUnicode_FromString`` to create a new ``PyObject*`` and returns it.

3. What about the following code

    .. code-block::

      const std::vector<float>& add(const std::vector<int>& a);

    - For the argument, it creates a ``type_caster<std::vector<int>>`` which contains a data member
      ``std::vector<int> value``. It assumes that the passed python object is a list and uses ``PyList_Check``
      to ensure that. ``type_caster<int>`` is used to parse the list and fill in ``value``.

    - For the return value, it converts a input ``std::vector<int>`` to a python list object. It does not
      matter whether it returns a ``const`` or a ``reference`` object.
