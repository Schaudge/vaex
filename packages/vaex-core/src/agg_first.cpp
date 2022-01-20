#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

// TODO: support DataType != DataType2
template <class DataType = double, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggFirstPrimitive : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    using data_type = DataType;
    using data_type2 = DataType2;

    AggFirstPrimitive(Grid<IndexType> *grid, int grids, int threads, bool invert)
        : Base(grid, grids, threads), data_ptr2(threads), data_size2(threads), data_mask_ptr2(threads), data_mask_size2(threads), invert(invert) {
        grid_data_order = new data_type2[this->count()];
        typedef std::numeric_limits<data_type2> limit_type;
        if (invert)
            std::fill(grid_data_order, grid_data_order + this->count(), limit_type::min());
        else
            std::fill(grid_data_order, grid_data_order + this->count(), limit_type::max());
    }
    virtual ~AggFirstPrimitive() { delete[] grid_data_order; }
    void set_data(int thread, py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (index == 1) {
            this->data_ptr2[thread] = (DataType2 *)info.ptr;
            this->data_size2[thread] = info.shape[0];
        } else {
            this->data_ptr[thread] = (DataType *)info.ptr;
            this->data_size[thread] = info.shape[0];
        }
    }
    void set_data_mask2(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr2[thread] = (uint8_t *)info.ptr;
        this->data_mask_size2[thread] = info.shape[0];
    }
    virtual void merge(std::vector<Aggregator *> others) {
        const bool invert = this->invert;
        for (auto i : others) {
            auto other = static_cast<AggFirstPrimitive *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                if (invert) {
                    if (other->grid_data_order[i] > this->grid_data_order[i]) {
                        this->grid_data[i] = other->grid_data[i];
                        this->grid_data_order[i] = other->grid_data_order[i];
                    }
                } else {
                    if (other->grid_data_order[i] < this->grid_data_order[i]) {
                        this->grid_data[i] = other->grid_data[i];
                        this->grid_data_order[i] = other->grid_data_order[i];
                    }
                }
            }
        }
    }
    virtual pybind11::object get_result() {
        const bool invert = this->invert;
        {
            py::gil_scoped_release release;
            for (size_t j = 0; j < this->grid->length1d; j++) {
                for (int64_t i = 1; i < this->grids; ++i) {
                    int64_t j2 = j + i * this->grid->length1d;
                    if (invert) {
                        if (grid_data_order[j2] > grid_data_order[j]) {
                            this->grid_data[j] = this->grid_data[j2];
                            grid_data_order[j] = grid_data_order[j2];
                        }
                    } else {
                        if (grid_data_order[j2] < grid_data_order[j]) {
                            this->grid_data[j] = this->grid_data[j2];
                            grid_data_order[j] = grid_data_order[j2];
                        }
                    }
                }
            }
        }
        py::object numpy = py::module::import("numpy");
        py::object self = py::cast(this);
        return numpy.attr("array")(self).attr("__getitem__")(0);
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        auto data_ptr = this->data_ptr[thread];
        auto data_ptr2 = this->data_ptr2[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto data_mask_ptr2 = this->data_mask_ptr2[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        auto grid_data_order = &this->grid_data_order[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        // if (data_ptr2 == nullptr) {
        //     throw std::runtime_error("data2 not set");
        // }
        const bool invert = this->invert;
        // TODO: masked support
        for (size_t j = 0; j < length; j++) {
            DataType value = data_ptr[offset + j];
            DataType2 value_order = data_ptr2 == nullptr ? offset + j : data_ptr2[offset + j];
            if (FlipEndian) {
                value = _to_native(value);
                value_order = _to_native(value_order);
            }
            if (value == value && value_order == value_order) { // nan check
                IndexType i = indices1d[j];
                if (invert) {
                    if (value_order > grid_data_order[i]) {
                        grid_data[i] = value;
                        grid_data_order[i] = value_order;
                    }
                } else {
                    if (value_order < grid_data_order[i]) {
                        grid_data[i] = value;
                        grid_data_order[i] = value_order;
                    }
                }
            }
        }
    }
    data_type2 *grid_data_order;

    std::vector<data_type2 *> data_ptr2;
    std::vector<uint64_t> data_size2;
    std::vector<uint8_t *> data_mask_ptr2;
    std::vector<uint64_t> data_mask_size2;
    bool invert; // intead of creating 2x as many templates
};               // namespace vaex

template <class T, class T2, bool FlipEndian>
void add_agg_first_primitive_mixed(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggFirst_");
    class_name += type_name<T>::value;
    class_name += "_";
    class_name += type_name<T2>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggFirstPrimitive<T, T2, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int, bool>(), py::keep_alive<1, 2>()).def_buffer(&Class::buffer_info).def("set_data_mask2", &Class::set_data_mask2);
    // .def("clear_data_mask2", &Class::clear_data_mask2);
}

template <class T, bool FlipEndian>
void add_agg_first_primitive(py::module &m, const py::class_<Aggregator> &base) {
#define create(type) add_agg_first_primitive_mixed<T, type, FlipEndian>(m, base);
#include "create_alltypes.hpp"
}

#undef create
#define create(type)                                                                                                                                                                                   \
    template void add_agg_first_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                             \
    template void add_agg_first_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex