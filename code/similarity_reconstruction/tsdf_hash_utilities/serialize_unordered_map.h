#pragma once
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <unordered_map>
#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization {

    template<class Archive, typename... TArgs >
        inline void save(Archive & ar, std::unordered_map<TArgs...> const&t, unsigned) {
            boost::serialization::stl::save_collection<Archive, std::unordered_map<TArgs...> >(ar, t);
        }

    template<class Archive, typename... TArgs >
        inline void load(Archive & ar, std::unordered_map<TArgs...> &t, unsigned) {
            boost::serialization::stl::load_collection<Archive,
                std::unordered_map<TArgs...>,
                boost::serialization::stl::archive_input_map<
                    Archive, std::unordered_map<TArgs...> >,
                boost::serialization::stl::no_reserve_imp<std::unordered_map<TArgs...> >
                    >(ar, t);
        }

    // split non-intrusive serialization function member into separate
    // non intrusive save/load member functions
    template <class Archive, typename... TArgs>
        inline void serialize(Archive & ar, std::unordered_map<TArgs...> &t, unsigned file_version) {
            boost::serialization::split_free(ar, t, file_version);
        }
} }

