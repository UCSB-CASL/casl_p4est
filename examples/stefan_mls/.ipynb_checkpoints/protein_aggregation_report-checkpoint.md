This section documents the modeling, implementation, and analysis of protein aggregation using a Stefan-type moving boundary solver, based on insights developed during discussions with Claude AI.

---

## Overview

Protein aggregation was modeled as a **diffusion-limited growth** problem with **patch-dependent interfacial kinetics**, adapting a solver originally designed for ice melting (e.g., Frank sphere) to simulate irreversible binding of monomers onto protein aggregates.  

The solver used a **single level-set approach** with **spatially varying boundary coefficients** to mimic patch behavior.

---

## Mathematical Model

### 1. **Protein Transport Equation (in Solution Region, $\phi < 0$)**

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot (D(x) \nabla \rho)
$$

- $\rho(x,t)$: protein concentration  
- $D(x)$: position-dependent diffusivity

### 2. **Laplace Equation (Inside Aggregate, $\phi > 0$)**

$$
\nabla^2 \rho = 0
$$

Assumes zero diffusion within the aggregate.

### 3. **Level Set Evolution Equation**

$$
\frac{\partial \phi}{\partial t} + v_n |\nabla \phi| = 0
$$

- $\phi(x,t)$: level set function (negative in solution, positive in aggregate)  
- $v_n$: interface normal velocity

### 4. **Jump Condition at Interface ($\phi = 0$)**

$$
v_n = \frac{k_s \nabla \rho_s \cdot \mathbf{n} - k_l \nabla \rho_l \cdot \mathbf{n}}{L \rho_l}
$$

- $k_s, k_l$: binding rates (patch-specific)  
- $L$: binding energy  
- $\rho_l$: solution density

### 5. **Robin Boundary Condition at Interface**

$$
\nabla \rho \cdot \mathbf{n} + \alpha \rho = \alpha \rho_{eq}
$$

- $\alpha$: binding coefficient (varies by patch)  
- $\rho_{eq}$: equilibrium concentration

---

## Implementation Details

### Key Variables

| Variable | Description |
|----------|-------------|
| `phi` | Level set function |
| `T_l` | Protein concentration in solution |
| `T_s` | Protein concentration in aggregate (fixed at 0) |
| `v_interface` | Interface velocity |

### Physical Parameters

```cpp
alpha_s = 0.0;     // No diffusion inside aggregates
alpha_l = 0.1;     // Diffusion in solution
k_s = 1.5;         // Binding rate for hydrophobic patches
k_l = 0.5;         // Binding rate for hydrophilic patches
L = 1.0;           // Binding energy
rho_l = 1.0;       // Solution density
Tinterface = 0.0;  // Equilibrium concentration at interface
Twall = 1.0;       // Concentration at boundaries
```

### Patch Implementation (C++ Snippet)

```cpp
bool is_hydrophobic_patch(DIM(double x, double y, double z)) {
  double r = sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));
  return (r < 0.21 || (r < 0.3 && x > 0.1));
}

bool is_hydrophilic_patch(DIM(double x, double y, double z)) {
  double r = sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));
  return (r < 0.3 && !is_hydrophobic_patch(DIM(x, y, z)));
}
```
### Velocity Modification Based on Patch Type
```cpp
bool hydrophobic = is_hydrophobic_patch(DIM(xyz[0], xyz[1], xyz[2]));
double growth_factor = hydrophobic ? 2.0 : 0.5;

foreach_dimension(d) {
  jump.ptr[d][n] *= growth_factor;
}
```

## Solver Workflow

**Initialize:**
- Create initial spherical protein aggregate  
- Set up concentration gradient from boundaries  

**For each timestep:**
- Extend fields across interface  
- Compute interfacial velocity based on concentration gradients  
- Advance level set function  
- Update grid based on new level set  
- Solve diffusion equation in both domains  
- Check for convergence  

**Visualization:**
- Monitor phase field ($\phi$)  
- Track concentration fields ($T_l$, $T_s$)  
- Visualize interface velocity  

---

## Results & Analysis

### Observations
- Initial growth phase with non-zero velocity  
- Eventual steady state with zero interface velocity  
- Uniform, circular growth pattern  
- Stable concentration gradient  

### Expected vs. Actual Behavior
The model successfully captured:
- ✓ Diffusion-driven growth  
- ✓ Zero diffusion inside aggregates  
- ✓ Concentration gradient from boundaries  
- ✓ Evolution to physically realistic steady state  

However, the patches did not produce visible anisotropy as expected:
- The growth remained uniformly circular  
- Patch effects were not strong enough to break symmetry  

### Interpretation
The simulation results align with diffusion-limited growth theory, showing:
- Initial rapid growth when concentration gradient is steep  
- Gradual slowing as nearby monomers are depleted  
- Eventual steady state when diffusion rate balances attachment/detachment rates  

The **uniform circular shape** suggests that patch effects need enhancement to create more realistic, **non-uniform** protein aggregates.
