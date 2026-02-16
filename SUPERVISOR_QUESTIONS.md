# Questions for Supervisor

## Network Design

1. **Is N=100 sufficient for the thesis?**
   We're currently using 100 nodes for fast iteration. Should we scale up to 500–1,000 for the final results, or is 100 acceptable for demonstrating the methodology?

2. **Are ER, BA, and Complete the right three networks?**
   - ER = random baseline
   - BA = scale-free (primary, realistic)
   - Complete = negative control (maximally symmetric)

   Should we add any other topology? (e.g., Watts–Strogatz small-world, lattice/grid, or a real-world network dataset?)

3. **ER edge probability p=0.05 — is this appropriate?**
   This gives ⟨k⟩ ≈ 4.5. Should p be tuned to match the BA average degree (⟨k⟩ ≈ 5.8) for a fairer comparison, or is the difference intentional?

---

## Diffusion Models

4. **Which diffusion model should be the primary focus?**
   We have IC, SI, and SIR implemented. For the thesis, should we:
   - Focus on one model in depth (which one?), or
   - Compare all three models as part of the analysis?

5. **Is the R₀ → parameter mapping correct?**
   We map target R₀ to transmission probability using:
   - IC: `p = R₀ / ⟨k⟩`
   - SI: `β = R₀ / ⟨k⟩`
   - SIR: `β = R₀ · γ / ⟨k⟩`

   Is this the standard approach, or should we use a different calibration method?

6. **R₀ values {0.5, 1.0, 1.5, 2.0, 3.0} — is this the right range?**
   Should we test more granular values around the critical threshold (R₀ = 1.0)?

7. **SIR recovery probability γ = 0.2 — is this a good default?**
   Should we also vary γ, or keep it fixed and only vary β?

---

## Source Detection (Next Phase)

8. **What ML approach do you recommend for source detection?**
   Options we're considering:
   - Graph Neural Networks (GNNs)
   - Centrality-based heuristics (e.g., rumour centrality, Jordan centre)
   - Classical ML on hand-crafted graph features
   - Some combination

   Do you have a preference or a specific paper we should follow?

9. **What is the "observed" input for the ML model?**
   Currently we store:
   - **Directed infection tree** (ground truth — not available in practice)
   - **Undirected observed graph** (just which nodes are infected + their connections)

   In a real scenario, would the model also know **infection times**, or only the final set of infected nodes?

10. **How should we evaluate source detection performance?**
    - Top-1 accuracy (exact match)?
    - Top-k accuracy?
    - Distance from true source in the graph?
    - All of the above?

---

## Scope & Thesis Structure

11. **Is the current scope appropriate for a bachelor's thesis?**
    Current pipeline: network generation → diffusion simulation → source detection.
    Is this too much / too little?

12. **Should we include a theoretical analysis alongside the computational experiments?**
    For example, proving or discussing why source detection is harder on symmetric networks (complete graph) vs. heterogeneous ones (BA).

13. **Are there specific papers we should cite or replicate?**
    Key references we're aware of:
    - Shah & Zaman (2011) — Rumour source detection
    - Pinto et al. (2012) — Network source detection with partial observations
    - Any others?

---

## Practical / Administrative

14. **When is the next progress checkpoint?**

15. **Any formatting or structure requirements for the thesis document?**
