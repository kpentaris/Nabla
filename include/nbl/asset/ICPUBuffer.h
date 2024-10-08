// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_

#include <type_traits>

#include "nbl/core/alloc/null_allocator.h"

#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPreHashed.h"

namespace nbl::asset
{

//! One of CPU class-object representing an Asset
/**
    One of Assets used for storage of large arrays, so that storage can be decoupled
    from other objects such as meshbuffers, images, animations and shader source/bytecode.

    @see IAsset
*/
class ICPUBuffer : public asset::IBuffer, public IPreHashed
{
    protected:
        //! Non-allocating constructor for CCustormAllocatorCPUBuffer derivative
        ICPUBuffer(size_t sizeInBytes, void* dat) : asset::IBuffer({ dat ? sizeInBytes : 0,EUF_TRANSFER_DST_BIT }), data(dat) {}

    public:
        //! Constructor. TODO: remove, alloc can fail, should be a static create method instead!
        /** @param sizeInBytes Size in bytes. If `dat` argument is present, it denotes size of data pointed by `dat`, otherwise - size of data to be allocated.
        */
        ICPUBuffer(size_t sizeInBytes) : asset::IBuffer({0,EUF_TRANSFER_DST_BIT})
        {
            data = _NBL_ALIGNED_MALLOC(sizeInBytes,_NBL_SIMD_ALIGNMENT);
            if (!data) // FIXME: cannot fail like that, need factory `create` methods
                return;

            m_creationParams.size = sizeInBytes;
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override final
        {
            auto cp = core::make_smart_refctd_ptr<ICPUBuffer>(m_creationParams.size);
            memcpy(cp->getPointer(), data, m_creationParams.size);
            return cp;
        }

        constexpr static inline auto AssetType = ET_BUFFER;
        inline IAsset::E_TYPE getAssetType() const override final { return AssetType; }

        inline size_t getDependantCount() const override {return 0;}

        //
        inline core::blake3_hash_t computeContentHash() const override
        {
			core::blake3_hasher hasher;
            if (data)
                hasher.update(data,m_creationParams.size);
			return static_cast<core::blake3_hash_t>(hasher);
        }

        inline bool missingContent() const override {return !data;}

        //! Returns pointer to data.
        const void* getPointer() const {return data;}
        void* getPointer() 
        { 
            assert(isMutable());
            return data;
        }
        
        inline core::bitflag<E_USAGE_FLAGS> getUsageFlags() const
        {
            return m_creationParams.usage;
        }
        inline bool setUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
        {
            assert(isMutable());
            m_creationParams.usage = _usage;
            return true;
        }
        inline bool addUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
        {
            assert(isMutable());
            m_creationParams.usage |= _usage;
            return true;
        }

    protected:
        inline IAsset* getDependant_impl(const size_t ix) override
        {
            return nullptr;
        }

        inline void discardContent_impl() override
        {
            return freeData();
        }

        // REMEMBER TO CALL FROM DTOR!
        // TODO: idea, make the `ICPUBuffer` an ADT, and use the default allocator CCPUBuffer instead for consistency
        // TODO: idea make a macro for overriding all `delete` operators of a class to enforce a finalizer that runs in reverse order to destructors (to allow polymorphic cleanups)
        virtual inline void freeData()
        {
            if (data)
                _NBL_ALIGNED_FREE(data);
            data = nullptr;
            m_creationParams.size = 0ull;
        }

        void* data;
};


template<
    typename Allocator = _NBL_DEFAULT_ALLOCATOR_METATYPE<uint8_t>,
    bool = std::is_same<Allocator, core::null_allocator<typename Allocator::value_type> >::value
>
class CCustomAllocatorCPUBuffer;

using CDummyCPUBuffer = CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true>;

//! Specialization of ICPUBuffer capable of taking custom allocators
/*
    Take a look that with this usage you have to specify custom alloctor
    passing an object type for allocation and a pointer to allocated
    data for it's storage by ICPUBuffer.

        So the need for the class existence is for common following tricks - among others creating an
        \bICPUBuffer\b over an already existing \bvoid*\b array without any \imemcpy\i or \itaking over the memory ownership\i.
        You can use it with a \bnull_allocator\b that adopts memory (it is a bit counter intuitive because \badopt = take\b ownership,
        but a \inull allocator\i doesn't do anything, even free the memory, so you're all good).
    */

template<typename Allocator>
class CCustomAllocatorCPUBuffer<Allocator,true> : public ICPUBuffer
{
        static_assert(sizeof(typename Allocator::value_type) == 1u, "Allocator::value_type must be of size 1");
    protected:
        Allocator m_allocator;

        virtual ~CCustomAllocatorCPUBuffer() final
        {
            freeData();
        }
        inline void freeData() override
        {
            if (ICPUBuffer::data)
                m_allocator.deallocate(reinterpret_cast<typename Allocator::pointer>(ICPUBuffer::data), ICPUBuffer::m_creationParams.size);
            ICPUBuffer::data = nullptr; // so that ICPUBuffer won't try deallocating
        }

    public:
        CCustomAllocatorCPUBuffer(size_t sizeInBytes, void* dat, core::adopt_memory_t, Allocator&& alctr = Allocator()) : ICPUBuffer(sizeInBytes,dat), m_allocator(std::move(alctr))
        {
        }
};

template<typename Allocator>
class CCustomAllocatorCPUBuffer<Allocator, false> : public CCustomAllocatorCPUBuffer<Allocator, true>
{
        using Base = CCustomAllocatorCPUBuffer<Allocator, true>;

    protected:
        virtual ~CCustomAllocatorCPUBuffer() = default;
        inline void freeData() override {}

    public:
        using Base::Base;

        // TODO: remove, alloc can fail, should be a static create method instead!
        CCustomAllocatorCPUBuffer(size_t sizeInBytes, const void* dat, Allocator&& alctr = Allocator()) : Base(sizeInBytes, alctr.allocate(sizeInBytes), core::adopt_memory, std::move(alctr))
        {
            memcpy(Base::data,dat,sizeInBytes);
        }
};

} // end namespace nbl::asset

#endif