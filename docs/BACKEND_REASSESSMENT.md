# Backend reassessment

## Question

Should Jengine continue using `ash` + Vulkan directly, or should it switch to a narrower internal compute abstraction and possibly a different backend surface?

## Current answer

**Keep `ash` + Vulkan as the execution backend for now.**

At the same time, continue to tighten the *internal* abstraction boundary so the rest of the runtime depends on a smaller compute-facing interface.

## Why keeping Vulkan still makes sense

### 1. The measured wins are already showing up at the kernel level

Real measurements on this machine show:

- Vulkan dense `q_proj` matvec: about `2.519 ms`
- Vulkan packed `q_proj` matvec: about `1.711 ms`

So the backend is already capable of exposing the packed-kernel advantage we care about.

### 2. The current broad hybrid slowdown is not caused by Vulkan itself

The main losses in broader hybrid decode have come from:

- pack/setup overhead
- compile/setup overhead
- cache reuse gaps
- multi-projection orchestration cost

Those are higher in the stack than raw Vulkan command submission.

### 3. Jengine needs unusual kernel control

This project is not just dispatching ordinary dense GEMMs. It needs:

- ternary-g128 packed weights
- mixed dense and packed execution
- reusable cached runners
- explicit upload / compute / download accounting
- custom projection-by-projection experimentation

That is exactly the kind of work where raw Vulkan control is useful.

### 4. The hardware fit is good

This machine has:

- modern Intel Lunar Lake graphics
- working Vulkan compute queues
- successful dense and packed Vulkan kernel bring-up

There is no evidence yet that the backend is the wrong low-level choice.

## What should change

The internal compute boundary should become narrower and cleaner.

Recommended layering:

1. runtime / decode scheduler
2. projection and tensor execution interface
3. Vulkan-backed implementation
4. shader / kernel modules

That gives us two benefits:

- less Vulkan-specific surface area in the runtime
- easier future experimentation if we ever test another backend

## When to reconsider

A backend change would be worth revisiting if one of these becomes true:

1. Vulkan blocks a needed optimization on this Intel stack
2. shader / pipeline complexity starts dominating iteration speed
3. a thinner internal abstraction is fully in place and another backend can be tested cheaply
4. a different backend shows clearly better reuse, compile latency, or integration on this hardware

## Recommendation

Near term:

- keep `ash` + Vulkan
- improve cache reuse and orchestration above it
- continue moving toward a smaller internal compute interface

Do **not** spend time replacing Vulkan right now. The current bottlenecks are still above the backend layer.
