using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using CSparse.Storage;
using CSparse.Double.Factorization;
using CSparse;
using C2M2.Utils;
using C2M2.NeuronalDynamics.UGX;

namespace C2M2.NeuronalDynamics.Simulation
{
    /// <summary>
    /// This is the sparse solver class for solving the Hodgkin-Huxley equations for the propagation of action potentials. Below are the HH equations
    /// this is a system of partial differential equations (PDE), the first equation is for the membrane potential (voltage) it is time spatially dependent
    /// and the remaining equations are the ODE equations for the state variables n,m,h (these are unitless). The system is non-trivial in that the PDE equation on Voltage
    /// is coupled to the ODE equations using n,m,h and these ODE equations are also non-linear.
    /// 
    ///\f[\frac{a}{2R}\frac{\partial^2V}{\partial x^2}=C\frac{\partial V}{\partial t}+\bar{g}_{K}n^4(V-V_k)+\bar{g}_{Na}m^3h(V-V_{Na})+\bar{g}_l(V-V_l)\f]
    ///
    ///\f[\frac{dn}{dt}=\alpha_n(V)(1-n)-\beta_n(V)n\f]
    ///
    ///\f[\frac{dm}{dt}=\alpha_m(V)(1-m)-\beta_m(V)m\f]
    ///
    ///\f[\frac{dh}{dt}=\alpha_h(V)(1-h)-\beta_h(V)h\f]
    ///
    /// The main solver begins with the function call <c>Solve()</c> prior to each iteration of <c>Solve()</c> the new voltage
    /// values <c>U</c> are sent to this class and then new <c>U</c> are sent out from <c>Solve()</c>
    /// 
    /// The solver currently uses SBDF2 (semi-implicit) Backward Difference 2
    /// The solver takes into account the non-uniform radii of the geometry and the non-uniform edgelength of the geometry
    /// 
    /// These parameters need to be made available to the user to modify for their particular simulation parameters
    /// The rate functions are defined at the end
    /// Note: Very important --> ALL UNITS FOR THE SOLVER ARE IN MKS, therefore when modifying the color bars ranges, and raycast/clamp hit values
    /// they need to be in [V] not [mV] so if you intend to use 50 [mV] it needs to be coded as 0.05 [V]
    /// 
    /// The simulation parameters are defined first, initial voltage hit value for raycasting, the endTime, time step size (k)
    /// Also other options are defined here such as having the SomaOn for clamping the soma on for voltage clamp tests (this is 
    /// mostly used for verifying the voltage output against Yale Neuron).
    /// </summary>

    //tex: Below are the Hodgkin Huxley equations
    //$$\frac{a}{2R}\frac{\partial^2V}{\partial x^2}=C\frac{\partial V}{\partial t}+\bar{g}_{K}n^4(V-V_k)+\bar{g}_{Na}m^3h(V-V_{Na})+\bar{g}_l(V-V_l)$$
    //$$\frac{dn}{dt}=\alpha_n(V)(1-n)-\beta_n(V)n$$
    //$$\frac{dm}{dt}=\alpha_m(V)(1-m)-\beta_m(V)m$$
    //$$\frac{dh}{dt}=\alpha_h(V)(1-h)-\beta_h(V)h$$

    public class SparseSolverTestv1 : NDSimulation
    {
        ///<summary>
        /// This is the voltage for the voltage clamp, this is primarily used for when we do the convergence analysis of the code using a 
        /// soma clamp at 50 [mV], the units for voltage in the solver is [V] that is why <c>vstart</c> is set to 0.05
        ///</summary>
        public double vstart = 0.050;         
        ///<summary>
        /// [ohm.m] resistance.length, this is the axial resistence of the neuron, increasing this value has the effect of making the AP waves more localized and slower conduction speed
        /// decreasing this value has the effect of make the AP waves larger and have a faster conduction speed
        /// </summary>
        private double res = 300.0 * 1.0E-2;
        /// <summary>
        /// [F/m2] capacitance per unit area, this is the plasma membrane capacitance, this a standard value for the capacitance
        /// </summary>
        private double cap = 1.0 * 1.0E-2;
        /// <summary>
        /// [S/m2] potassium conductance per unit area, this is the Potassium conductance per unit area, it is used in this term
        /// \f[\bar{g}_{K}n^4(V-V_k)\f]
        /// where \f$n\f$ is the state variable, and \f$V_k\f$ is the reversal potential.
        /// </summary>
        // 50, not 60: matches meshgenerator's IonChannelModels.cs and
        // NeuroVISOR-CSharpStudies' SimParams.Gk (paper-validated) - rosado's
        // old value of 60 produced a conduction speed persistently ~2.6% too
        // fast vs. NEURON even at the finest timestep (not a temporal-order
        // issue - a model-constant mismatch), found via the coarsening study.
        private double gk = 5.0 * 1.0E1;
        /// <summary>
        /// [S/m2] sodium conductance per unit area, this is the Sodium conductance per unit area, it is used in this term
        /// \f[\bar{g}_{Na}m^3h(V-V_{Na})\f]
        /// where \f$m,h\f$ are the state variables, and \f$V_{Na}\f$ is the reversal potential for sodium.
        /// </summary>
        // 50, not 56: same fix as gk above (matches meshgenerator/SimParams.Gna).
        private double gna = 50.0 * 1.0E1;
        
        /// <summary>
        /// [S/m2] leak conductance per unit area, this is the leak conductance per unit area, it is used in this term
        /// \f[\bar{g}_{l}(V-V_l)\f]
        /// \f$V_l\f$ is the leak reversal potential.
        /// </summary>
        private double gl = 0.1 * 1.0E1;
        /// <summary>
        /// [V] potassium reversal potential
        /// </summary>
        private double ek = -90.0 * 1.0E-3;

        /// <summary>
        /// [V] sodium reversal potential
        /// </summary>
        private double ena = 50.0 * 1.0E-3;

        /// <summary>
        /// [V] leak reversal potential
        /// </summary>
        private double el = -70.0 * 1.0E-3;
        /// <summary>
        /// [V] threshold shift V_T of the rate functions below, written in terms
        /// of V - V_T. -55 mV matches the paper and NeuroVISOR-CSharpStudies'
        /// SimParams.VTraub (meshgenerator's IonChannelModels.vTraubMv, same value).
        /// </summary>
        private static double Vt = -55.0 * 1.0E-3;
        /// <summary>
        /// [] potassium channel state probability, unitless. Computed in
        /// InitializeNeuronCell as alpha_n/(alpha_n+beta_n) at V = VStart, so the
        /// cell always starts at its own true resting equilibrium instead of a
        /// value baked in for the old (wrong) Vt.
        /// </summary>
        private double ni;
        /// <summary>
        /// [] sodium channel state probability, unitless. Computed in
        /// InitializeNeuronCell as alpha_m/(alpha_m+beta_m) at V = VStart.
        /// </summary>
        private double mi;
        /// <summary>
        /// [] sodium channel state probability, unitless. Computed in
        /// InitializeNeuronCell as alpha_h/(alpha_h+beta_h) at V = VStart.
        /// </summary>
        private double hi;
        /// <summary>
        /// These are the solution vectors for the voltage <code>U</code>
        /// the state <c>M</c>, state <c>N</c>, and state <c>H</c>
        /// </summary>
        private Vector U;
        /// <summary>
        /// This is the U that gets modified during the step before U is set to it.
        /// </summary>
        private Vector U_Active;
        /// <summary>
        /// These are the solution vectors for the voltage <code>U</code>
        /// the state <c>M</c>, state <c>N</c>, and state <c>H</c>
        /// </summary>
        private Vector M, N, H;
        /// <summary>
        /// This is for the synaptic current. It contains:
        ///     [0]: The current at the active time step.
        ///     [1]: The current at the previous time step.
        /// </summary>
        private List<Vector> Isyn;
        /// <summary>
        /// The spatial scaling at post-synaptic location for time stepping (1/(cap * area))
        /// </summary>
        private Vector surfaceArea;
        /// <summary>
        /// this is for storing previous states
        /// </summary>
        private Vector Upre, Npre, Mpre, Hpre;
        /// <summary>
        /// This is a vector the Reaction terms
        /// </summary>
        private Vector R;       
        /// <summary>
        /// This is an array for the right hand side of the problem Ax = b
        /// </summary>
        private double[] b;                               
        /// <summary>
        /// Temporary state vector
        /// </summary>
        private Vector tempState;

        List<double> reactConst;                            //This is for passing the reaction function constants
        List<CoordinateStorage<double>> sparse_stencils;
        CompressedColumnStorage<double> r_csc;              //This is for the rhs sparse matrix
        CompressedColumnStorage<double> l_csc;              //This is for the lhs sparse matrix
        private SparseLU SBDF_implicit_decomp;                                //Initialize the LU factorizaation
        /// <summary>
        /// SBDF2 needs two past voltage/gating levels, but at t=0 only V^0
        /// exists; starting it with V^(-1)=V^0 instead makes the whole run
        /// first order in time (paper Sec. 3.1.2). This is the factorization of
        /// I - timeStep*A used only for SolveStep's t==0 IMEX-Euler start-up
        /// step - makeSparseStencils already scales its matrix by (2/3)*its dt
        /// argument, so building it with 1.5*timeStep (in PreSolve) gives
        /// exactly I - timeStep*A, matching NeuroVISOR-CSharpStudies'
        /// Sbdf2Solver.cs FactorStartup/SolveStartup.
        /// </summary>
        private SparseLU luStartup;
        /// <summary>
        /// The matrix I - timeStep*A that <see cref="luStartup"/> factors; its rows are needed by
        /// the Dirichlet updates after the start-up step.
        /// </summary>
        private CompressedColumnStorage<double> startupLcsc;
        /// <summary>
        /// True between the start-up step (t == 0) and the next step: the Dirichlet updates in
        /// PostSolveStep(0) must modify the system that step solved, I - timeStep*A with the
        /// IMEX-Euler right-hand side, not the SBDF2 one.
        /// </summary>
        private bool startupStepSolved = false;

        /// <summary>
        /// Send simulation 1D values, this send the current voltage after the solve runs 1 iteration
        /// it passes <c>curVals</c>
        /// </summary>
        /// <returns>curVals</returns>
        public override double[] Get1DValues()
        {
            /// this initialize the curVals which will be sent back to the VR simulation
            double[] curVals = null;
            /// check if this beginning of the simulation
            if (currentTimeStep > -1)
            {
                Vector curTimeSlice;
                lock (visualizationValuesLock)
                {
                    /// define the current time slice to send and initialize it to the correct size which is the number of vertices in the geometry
                    /// initialize it to the current state of the voltage, this is the voltage we are sending back to vr simulation
                    curTimeSlice = U.SubVector(0, Neuron.nodes.Count);
                }
                //curTimeSlice.Multiply(1, curTimeSlice);

                curVals = curTimeSlice.ToArray();
            }
            return curVals;
        }

        /// <summary>
        /// Receive new simulation 1D index/value pairings
        /// Carefully, notice that <c>val</c> needs to be multiplied by 0.001 this is because
        /// the hit value is in [mV] and the solver uses [V]
        /// </summary>
        /// <param name="newValues"></param>
        public override void Set1DValues((int, double)[] newValues)
        {
            // De-duplicate by vertex (last entry wins, matching the old per-call-overwrite
            // behavior if the same vertex appears twice in one batch), while preserving
            // insertion order so a later duplicate still wins predictably.
            var valid = new Dictionary<int, double>();
            foreach ((int, double) newVal in newValues)
            {
                if (newVal.Item1 >= 0 && newVal.Item1 < Neuron.nodes.Count)
                {
                    valid[newVal.Item1] = newVal.Item2;
                }
            }
            if (valid.Count == 0) return;

            Vector result;
            if (valid.Count == 1)
            {
                // single Dirichlet row: the plain Sherman-Morrison rank-1 solve is already
                // exact on its own, no need for the more general rank-r machinery below.
                var only = valid.First();
                result = DircheletRank1UpdateSolve((only.Key, only.Value));
            }
            else
            {
                // r >= 2 simultaneous Dirichlet rows (multiple clamps, multiple direct-
                // stimulation hits, or both together): solve them all AT ONCE via the
                // Sherman-Morrison-Woodbury rank-r update (paper Sec. 6.3, eqn:woodbury),
                // so every one of them is exact - not just the last one applied (item 18).
                result = DircheletRankRUpdateSolve(valid);
            }

            lock (visualizationValuesLock)
            {
                U_Active = Vector.Build.DenseOfVector(result);
            }
        }

        /// <summary>
        /// this perform a Rank1UpdateSolve to properly adjust for added dirichelet boundary conditions.
        /// </summary>
        /// <param name="newVal"></param>
        /// <returns></returns>
        public Vector DircheletRank1UpdateSolve((int, double) newVal)
        {
            double[] bj = new double[Neuron.nodes.Count];
            double[] z = new double[Neuron.nodes.Count];
            double[] y = new double[Neuron.nodes.Count];
                        
            Vector ej = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
            Vector rj = ej.Clone();
            Vector ZZ = ej.Clone();
            Vector YY = ej.Clone();

            // the system the last step solved: the start-up one after t == 0, else SBDF2
            SparseLU decomp = startupStepSolved ? luStartup : SBDF_implicit_decomp;
            CompressedColumnStorage<double> lhs = startupStepSolved ? startupLcsc : l_csc;

            R.At(newVal.Item1, newVal.Item2);
            ej = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
            ej.At(newVal.Item1, 1.0);
            (lhs.Transpose()).Multiply(ej.ToArray(), bj);
            rj = Vector.Build.DenseOfArray(bj);
            rj.At(newVal.Item1, rj[newVal.Item1] - 1);

            decomp.Solve(ej.ToArray(), z);
            decomp.Solve(R.ToArray(), y);
            
            ZZ = Vector.Build.DenseOfArray(z);
            YY = Vector.Build.DenseOfArray(y);

            return YY.Add(ZZ.Multiply(rj.DotProduct(YY) / (1 - rj.DotProduct(ZZ))));
        }

        /// <summary>
        /// Rank-r generalization of <see cref="DircheletRank1UpdateSolve"/>: imposes Dirichlet
        /// boundary conditions at r &gt;= 2 distinct vertices SIMULTANEOUSLY and exactly, via the
        /// Sherman-Morrison-Woodbury formula (paper Sec. 6.3, eqn:woodbury) - fixes item 18
        /// ("more than one simultaneous clamp is not imposed correctly"): calling the rank-1
        /// solve once per vertex only leaves the LAST one exact, because each call re-derives
        /// its correction from the original (unclamped) matrix M, so any earlier row
        /// replacement is forgotten.
        ///
        /// Letting M' be M with rows k_1..k_r each replaced by an identity row e_{k_i}^T:
        ///   M' = M - E C^T,  E = [e_{k_1} .. e_{k_r}],  C[:,i] = (row k_i of M)^T - e_{k_i}
        /// Woodbury:
        ///   x = M'^{-1} R = y + Z (I_r - C^T Z)^{-1} C^T y,  where Z = M^{-1} E,  y = M^{-1} R
        /// which needs r+1 solves against the existing LU factorization of M (one per clamped
        /// vertex's unit column, plus one for R), plus one small r x r dense solve - no
        /// refactorization of M itself.
        /// </summary>
        /// <param name="newVals">vertex -> target Dirichlet voltage, r >= 2 distinct vertices</param>
        public Vector DircheletRankRUpdateSolve(IReadOnlyDictionary<int, double> newVals)
        {
            int n = Neuron.nodes.Count;
            int r = newVals.Count;
            int[] idx = newVals.Keys.ToArray();

            // the system the last step solved: the start-up one after t == 0, else SBDF2
            SparseLU decomp = startupStepSolved ? luStartup : SBDF_implicit_decomp;
            CompressedColumnStorage<double> lhs = startupStepSolved ? startupLcsc : l_csc;

            // R is the shared reaction RHS vector; stamp every clamped vertex's target
            // voltage into it, same role as the single-clamp case's R.At(j, v).
            foreach (var kv in newVals) R.At(kv.Key, kv.Value);

            // C[:, i] = (row idx[i] of M)^T - e_{idx[i]}, and Z[:, i] = M^{-1} e_{idx[i]} -
            // the same "rj"/"z" vectors DircheletRank1UpdateSolve computes for a single
            // vertex, just one column per clamped vertex here.
            Matrix C = Matrix.Build.Dense(n, r);
            Matrix Z = Matrix.Build.Dense(n, r);
            double[] ejArr = new double[n];
            double[] rowI = new double[n];
            double[] zi = new double[n];
            for (int i = 0; i < r; i++)
            {
                System.Array.Clear(ejArr, 0, n);
                ejArr[idx[i]] = 1.0;

                (lhs.Transpose()).Multiply(ejArr, rowI);
                rowI[idx[i]] -= 1.0;
                C.SetColumn(i, rowI);

                decomp.Solve(ejArr, zi);
                Z.SetColumn(i, zi);
            }

            double[] yArr = new double[n];
            decomp.Solve(R.ToArray(), yArr);
            Vector y = Vector.Build.DenseOfArray(yArr);

            // S = I_r - C^T Z  (r x r, dense);  rhs = C^T y  (r-vector)
            Matrix S = Matrix.Build.DenseIdentity(r) - C.TransposeThisAndMultiply(Z);
            Vector rhs = C.TransposeThisAndMultiply(y);
            Vector alpha = S.Solve(rhs);

            // x = y + Z*alpha
            return y.Add(Z.Multiply(alpha));
        }

        /// <summary>
        /// Receives 1D information for synaptic communication
        /// newValues = is a list of (presynapse, postsynapse)
        /// </summary>
        /// <param name="newValues"></param>
        internal override void SetSynapseCurrent(List<(Synapse,Synapse)> newValues)
        {
            List<double> tmp = new List<double>();

            // iterate through teach (pre,post) synapse pair
            foreach ((Synapse,Synapse) newVal in newValues)
            {
                if ((newVal.Item1 != null) && (newVal.Item2 != null))
                {
                    // Membrane area [um^2] at the postsynaptic location, using the same control-volume
                    // membrane length the matrix itself uses at that node (item 14) instead of
                    // the global target edge length.
                    double areaMicrons2 = 2 * System.Math.PI * Neuron.nodes[newVal.Item2.FocusVert].NodeRadius * MembraneLengthMicrons(newVal.Item2.FocusVert);

                    //tmp[0] is current synaptic state, and tmp[1] is previous synaptic state
                    tmp = SynapseCurrentFunction(newVal, newVal.Item1.currentModel.Value, areaMicrons2);
                    Isyn[0][newVal.Item2.FocusVert] += tmp[0];
                    Isyn[1][newVal.Item2.FocusVert] += tmp[1];
                    // we have the neuron ID, the vertex ID on that neuron, and the current coming from that vertex
                    newVal.Item2.currentIsyn = new List<double> {newVal.Item1.simulation.simID, newVal.Item1.FocusVert, tmp[0]};
                    surfaceArea[newVal.Item2.FocusVert] = 1 / (cap * areaMicrons2 * 1e-12);
                }
            }
        }

        /// <summary>
        /// [V] Spike-detection threshold (paper Sec. 3.1.5) - re-checked against the
        /// -70 mV/-55 mV frame (items 1-2): still comfortably inside this model's spike peak (item 13).
        /// </summary>
        private const double spikeThreshold = 0.038;

        /// <summary>
        /// [nS/um^2] Largest synaptic conductance per membrane area of the synapse's node for which the
        /// explicit synaptic term stays stable at dt = 32 us (paper Sec. 2.8, Table tab:gaba_stability:
        /// 8 nS on the 0.205 um^2 distal endpoint is the largest stable value, 11.3 nS is unstable).
        /// The bound grows like dt^(-1/2).
        /// </summary>
        private const double maxConductancePerArea32us = 39.0;
        /// <summary>
        /// Post synapses whose conductance has already been reported as capped.
        /// </summary>
        private readonly HashSet<Synapse> cappedSynapses = new HashSet<Synapse>();

        /// <summary>
        /// Upward crossings of the spike-detection threshold at the presynaptic nodes on this cell
        /// between V^t (Upre) and V^(t+1) (U_Active). The activation time is the first time level
        /// at or above the threshold, t_s = t_(t+1) (paper Sec. 3.1.5). One activation per
        /// crossing (item 12); a later crossing re-activates.
        /// </summary>
        internal override void DetectSynapseActivations(List<Synapse> preSynapses, int t)
        {
            int level = t + 1;
            foreach (Synapse pre in preSynapses)
            {
                int k = pre.FocusVert;
                if (k < 0 || k >= Neuron.nodes.Count) continue;
                if ((U_Active[k] >= spikeThreshold) && (Upre[k] < spikeThreshold))
                {
                    pre.RecordActivation(level * timeStep, level);
                }
            }
        }

        /// <summary>
        /// This is the synaptic current function
        /// the input is a tuple (presyn, postsyn) = (item1, item2) respectively, and the membrane
        /// area of the postsynaptic node in um^2.
        /// Called from PreSolveStep(t), before the step from V^t to V^(t+1): returns the synaptic
        /// current at level t (V^t, t_t) and at level t-1 (V^(t-1), t_(t-1)), the B^i and B^(i-1)
        /// of the SBDF2 step.
        /// </summary>
        /// <param name="newVal"></param>
        /// <returns></returns>
        public List<double> SynapseCurrentFunction((Synapse, Synapse) newVal, ISynapseModel model, double areaMicrons2)
        {
            //List contains the current synaptic current at index 0 and previous synaptic current at index 1
            List<double> Icurrs = new List<double>();

            // Postsynaptic voltage: b(V)*(V-E) is evaluated at the synapse's OWN (postsynaptic)
            // node, per the paper and NeuroVISOR-CSharpStudies' NmdaSynapse.cs/GabaSynapse.cs
            // (item 9). The postsynaptic node is on this cell, so V^t and V^(t-1) are this
            // solver's own U_Active and Upre - not the visualization copy, which still holds the
            // previous step until SetOutputValues (item 17).
            int k = newVal.Item2.FocusVert;
            double postsynVoltage = U_Active[k];
            double postsynVoltagePrev = Upre[k];

            int level = currentTimeStep;
            double tCur = level * timeStep;
            double tPrev = (level - 1) * timeStep;
            // Activation detected by the presynaptic cell up to this level (DetectSynapseActivations).
            double ts = newVal.Item1.ActivationTimeAt(level);

            //Adds the synaptic currents for the current and previous timesteps
            Icurrs.Add(model.getModelCurrent(postsynVoltage, tCur, ts));
            Icurrs.Add(model.getModelCurrent(postsynVoltagePrev, tPrev, ts));

            // Stability of the explicit synaptic term: cap the conductance by the membrane area
            // of the postsynaptic node (paper Sec. 2.8).
            double gMax = maxConductancePerArea32us * System.Math.Sqrt(32e-6 / timeStep) * areaMicrons2 * 1e-9;
            double g = model.getConductance();
            if (g > gMax)
            {
                Icurrs[0] *= gMax / g;
                Icurrs[1] *= gMax / g;
                if (cappedSynapses.Add(newVal.Item2))
                {
                    GameManager.instance.DebugLogSafe(model.getModelName() + " synapse at node " + k + ": conductance "
                        + (g * 1e9).ToString("G3") + " nS exceeds the stability limit "
                        + (gMax * 1e9).ToString("G3") + " nS for its membrane area (" + areaMicrons2.ToString("G3")
                        + " um^2); capped to the limit.");
                }
            }

            // Only guard against a genuine computational error (NaN/Infinity), not a
            // legitimately large current (item 15).
            if (Double.IsNaN(Icurrs[0]) || Double.IsNaN(Icurrs[1]) || Double.IsInfinity(Icurrs[0]) || Double.IsInfinity(Icurrs[1]))
            {
                Icurrs[0] = 0.0; Icurrs[1] = 0.0;
            }

            return Icurrs;
        }

        /// <summary>
        /// Control-volume membrane length at a node: half the sum of its incident
        /// edge lengths, in the same (unscaled) units as Neuron.TargetEdgeLength -
        /// matches makeSparseStencils' avgEdgeLengths (item 6/14).
        /// </summary>
        private double MembraneLengthMicrons(int nodeIndex)
        {
            double sum = 0.0;
            foreach (var kv in Neuron.nodes[nodeIndex].AdjacencyList)
            {
                sum += kv.Value;
            }
            return 0.5 * sum;
        }

        /// <summary>
        /// This is a small routine call to initialize the Neuron Cell
        /// this will initialize the solution vectors which are <c>U</c>, <c>M</c>, <c>N</c>, and <c>H</c>
        /// </summary>
        protected override void PreSolve()
        {
            GameManager g = GameManager.instance;
            // if loading, the values from file will be set in BuildVectors and Set1DValues
            if (!g.Loading) InitializeNeuronCell();
            else BuildVectors(g.U, g.M, g.N, g.H, g.Upre, g.Mpre, g.Npre, g.Hpre);

            ///<c>R</c> this is the reaction vector for the reaction solve
            R = Vector.Build.Dense(Neuron.nodes.Count);            
            
            tempState = Vector.Build.Dense(Neuron.nodes.Count, 0);
            ///<c>reactConst</c> this is a small list for collecting the conductances and reversal potential which is sent to the reaction solve routine
            reactConst = new List<double> { gk, gna, gl, ek, ena, el };

            /// this sets the target time step size
            timeStep = SetTargetTimeStep(cap, 2 * Neuron.MaxRadius,2*Neuron.MinRadius, Neuron.TargetEdgeLength, gna, gk,gl, res, 1.0);
            ///UnityEngine.Debug.Log("Target Time Step = " + timeStep);

            ///<c>List<CoordinateStorage<double>> sparse_stencils = makeSparseStencils(Neuron, res, cap, k);</c> Construct sparse RHS and LHS in coordinate storage format, no zeros are stored \n
            /// <c>sparse_stencils</c> this is a list which contains only two matrices the LHS and RHS matrices for the Crank-Nicolson solve
            sparse_stencils = makeSparseStencils(Neuron, res, cap, timeStep);
            ///<c>CompressedColumnStorage</c> call Compresses the sparse matrices which are stored in <c>sparse_stencils[0]</c> and <c>sparse_stencils[1]</c>
            r_csc = CompressedColumnStorage<double>.OfIndexed(sparse_stencils[0]); //null;
            l_csc = CompressedColumnStorage<double>.OfIndexed(sparse_stencils[1]); //null;
            ///<c>double [] b</c> we define storage for the diffusion solve part
            b = new double[Neuron.nodes.Count];
            ///<c>var lu = SparseLU.Create(l_csc, ColumnOrdering.MinimumDegreeAtA, 0.1);</c> this creates the LU decomposition of the HINES matrix which is defined by <c>l_csc</c>
            SBDF_implicit_decomp = SparseLU.Create(l_csc, ColumnOrdering.MinimumDegreeAtA, 0.1);

            /// IMEX-Euler start-up factorization I - timeStep*A for SolveStep's
            /// t==0 branch - see luStartup's doc comment.
            var startupStencils = makeSparseStencils(Neuron, res, cap, 1.5 * timeStep);
            startupLcsc = CompressedColumnStorage<double>.OfIndexed(startupStencils[1]);
            luStartup = SparseLU.Create(startupLcsc, ColumnOrdering.MinimumDegreeAtA, 0.1);
        }

        /// <summary>
        /// This is the main solver, it is running on it own thread.
        /// The solver using SBDF2 for time steping, the implicit part is used for the diffusion and the explicit
        /// is for the reaction terms and state variables
        /// </summary>     
        protected override void SolveStep(int t)
        {
            startupStepSolved = (t == 0);
            if (t == 0)
            {
                // IMEX-Euler (SBDF1) start-up step - see luStartup's doc comment.
                // Forward Euler for the gating variables and voltage:
                // (I - timeStep*A) V^1 = V^0 + timeStep*(reaction + synaptic),
                // solved against luStartup instead of the regular SBDF2 matrix,
                // since SBDF2 needs a V^(-1) that does not exist yet at t=0.
                R = U_Active.Clone();
                R.Add(reactF(reactConst, U_Active, N, M, H, cap).Multiply(timeStep), R);
                // Subtracted: Isyn is an outward-positive current, same convention reactF
                // uses for the Na/K/leak currents (item 10) - an outward current should
                // DECREASE voltage, not increase it.
                R.Subtract(Isyn[0].PointwiseMultiply(surfaceArea).Multiply(timeStep), R);
                Isyn[0].Multiply(0.0, Isyn[0]);
                Isyn[1].Multiply(0.0, Isyn[1]);
                surfaceArea.Multiply(0.0, surfaceArea);

                luStartup.Solve(R.ToArray(), b);

                tempState = N.Clone();
                N.Add(fS(N, an(U_Active), bn(U_Active), timeStep).Multiply(timeStep), N);
                Npre = tempState.Clone();

                tempState = M.Clone();
                M.Add(fS(M, am(U_Active), bm(U_Active), timeStep).Multiply(timeStep), M);
                Mpre = tempState.Clone();

                tempState = H.Clone();
                H.Add(fS(H, ah(U_Active), bh(U_Active), timeStep).Multiply(timeStep), H);
                Hpre = tempState.Clone();

                N.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), N);
                M.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), M);
                H.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), H);

                Upre = U_Active.Clone();
                U_Active.SetSubVector(0, Neuron.nodes.Count, Vector.Build.DenseOfArray(b));
                return;
            }

            R = U_Active.Clone();
            explicitUpdate(R, Upre, reactF(reactConst, U_Active, N, M, H, cap), reactF(reactConst, Upre, Npre, Mpre, Hpre, cap), timeStep, 1);

            // Rsyn = (2/3)*timeStep*surfaceArea.*(2*Isyn[0] - Isyn[1]). Spre must be a separate zero
            // vector: passing Rsyn as both S and Spre made explicitUpdate read the half-updated S
            // and give (8/9)Isyn[0] - (2/3)Isyn[1], a synaptic current 3x too weak.
            var Rsyn = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
            explicitUpdate(Rsyn, Vector.Build.Dense(Neuron.nodes.Count, 0.0), Isyn[0], Isyn[1], timeStep, surfaceArea);
            // reset synaptic source this ensures that when you remove the synapse that Isyn becomes 0; 
            // therefore, current is not being sent to postsynapse once synapse is removed
            Isyn[0].Multiply(0.0, Isyn[0]);
            Isyn[1].Multiply(0.0, Isyn[1]);
            surfaceArea.Multiply(0.0, surfaceArea);
            // Subtracted: Isyn/Rsyn is an outward-positive current, same convention
            // reactF uses for the Na/K/leak currents (item 10) - an outward current
            // should DECREASE voltage, not increase it.
            R.Subtract(Rsyn, R);
            SBDF_implicit_decomp.Solve(R.ToArray(), b);

            tempState = N.Clone();
            explicitUpdate(N, Npre, fS(N, an(U_Active), bn(U_Active), timeStep), fS(Npre, an(Upre), bn(Upre), timeStep), timeStep, 1);
            Npre = tempState.Clone();

            tempState = M.Clone();
            explicitUpdate(M, Mpre, fS(M, am(U_Active), bm(U_Active), timeStep), fS(Mpre, am(Upre), bm(Upre), timeStep), timeStep, 1);
            Mpre = tempState.Clone();

            tempState = H.Clone();
            explicitUpdate(H, Hpre, fS(H, ah(U_Active), bh(U_Active), timeStep), fS(Hpre, ah(Upre), bh(Upre), timeStep), timeStep, 1);
            Hpre = tempState.Clone();

            N.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), N);
            M.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), M);
            H.Map(x => System.Math.Max(0.0, System.Math.Min(1.0, x)), H);


            Upre = U_Active.Clone();
            U_Active.SetSubVector(0, Neuron.nodes.Count, Vector.Build.DenseOfArray(b));
        }

        internal override void SetOutputValues()
        { lock (visualizationValuesLock) U = U_Active.Clone(); }
		
        /// <summary>
        /// This function sets the target time step size, below is the formula for the conduction speed of the action potential (wave speed)
        ///
        /// \f[v = \frac{1}{C}\sqrt{\frac{d}{R_a R_{mem}}}]
        ///
        /// where
        /// \f[R_{mem} = \frac{1}{g_{Na}m^3 h + g_K n^4}]
        /// 
        /// and n,m,h are the probability states which vary in time and depend on the voltage
        /// Notice that
        /// 
        /// \f[\frac{1}{R_{mem}}\leq g_{Na}+g_K=G_{mem-theor-max}]
        /// 
        /// therefore we define the max conduction speed vmax as
        /// 
        /// v_{max} = \frac{1}{C}\sqrt{\frac{d_max\cdot (g_{Na}+g_K)}{R_a}}
        /// 
        /// then we solve for our target \f[Delta t] by computing
        /// 
        /// \f[\Delta t = \frac{\Delta x}{v_{max}}]
        /// 
        /// where \f[Delta x] is the median edge length, in this case we use the average this is because our geometries are regularize and there
        /// are not excessively small edges in the graph geometry.
        ///  
        /// </summary>
        /// <param name="cap"></param> this is the capacitance
        /// <param name="maxDiameter"></param> this is the maximum diameter
        /// <param name="edgeLength"></param> this is the target edge length of the graph geometry
        /// <param name="gna"></param> this is the sodium conductance
        /// <param name="gk"></param> this is the potassium conductance
        /// <param name="res"></param> this is the axial resistance
        /// <param name="Rmemscf"></param> this is membrane resistance scale factor, since this is only a fraction of theoretical maximum
        /// <returns></returns>
        public static double SetTargetTimeStep(double cap, double maxDiameter, double minDiameter,double edgeLength ,double gna, double gk, double gl, double res, double cfl)
        {
            /// here we set the minimum time step size and maximum time step size
            /// the dtmin is based on prior numerical experiments that revealed that for each refinement level the 
            /// voltage profiles were visually accurate when compared to Yale Neuron for delta t at least 2 microseconds
            /// we want to avoid using dtmin; therefore I compute the upper bound (and lower bound for reference)
            //double dtmin = 2e-6;
            // Paper Sec. 2.6.1: SBDF2 diverges on the test cell at dt >= 50us;
            // dt <= 32us is recommended. This is the only place dt is actually
            // decided (this function's result unconditionally overwrites whatever
            // the loader otherwise defaulted to), so 32us is set here, once.
            double dtmax = 32e-6;
            double dt;

            double gll = gl; double scf = 1E-6; // to convert to micrometer of edgelengths and radii don't forget this!!!!

            // what happens if the leak conductance is 0
            if (gll == 0.0) { gll = 1.0; }
            
            double upper_bound = cap * edgeLength*scf * System.Math.Sqrt(res / (gll*minDiameter*scf));
            //double lower_bound = cap * edgeLength*scf * System.Math.Sqrt(res / (gna + gk + gl) * maxDiameter*scf);
            //GameManager.instance.DebugLogSafe("upper_bound = " + upper_bound.ToString());

            // some cells may have an upper bound that is too large for the solver, so choose the smaller of the two dtmax or upper_bound
            dt = System.Math.Min(upper_bound,dtmax);
            //GameManager.instance.DebugLogSafe("lower_bound = " + lower_bound.ToString());
            return dt;       
        }

        /// <summary>
        /// This function initializes the voltage vector <c>U</c> and the state vectors
        /// <c>M</c>, <c>N</c>, and <c>H</c> \n
        /// The input <c>Neuron.vertCount</c> is the vertex count of the neuron geometry \n
        /// <c>U</c> is initialized to VStart = -70 [mV] for the entire cell \n
        /// <c>M</c> is initialized to \f$m_i\f$ which is set by <c>mi</c> (steady state at VStart) \n
        /// <c>N</c> is initialized to \f$n_i\f$ which is set by <c>ni</c> \n
        /// <c>H</c> is initialized to \f$h_i\f$ which is set by <c>hi</c>
        /// </summary>
        private void InitializeNeuronCell()
        {
            const double VStart = -70.0 * 1.0E-3;

            lock (visualizationValuesLock)
            {
                U = Vector.Build.Dense(Neuron.nodes.Count, VStart);
                U_Active = U.Clone();
            }
            Upre = U_Active.Clone();

            // Gating variables start at their steady-state probability at VStart
            // (alpha/(alpha+beta)), so the cell starts at its own true resting
            // equilibrium instead of a value hardcoded for a different V_T.
            Vector vStartVec = Vector.Build.Dense(1, VStart);
            double an0 = an(vStartVec)[0], bn0 = bn(vStartVec)[0];
            double am0 = am(vStartVec)[0], bm0 = bm(vStartVec)[0];
            double ah0 = ah(vStartVec)[0], bh0 = bh(vStartVec)[0];
            ni = an0 / (an0 + bn0);
            mi = am0 / (am0 + bm0);
            hi = ah0 / (ah0 + bh0);

            M = Vector.Build.Dense(Neuron.nodes.Count, mi);
            N = Vector.Build.Dense(Neuron.nodes.Count, ni);
            H = Vector.Build.Dense(Neuron.nodes.Count, hi);

            Isyn = new List<Vector>();
            for (int i = 0; i < 2; i++)
            {
                // Create a dense vector for each node
                var synVector = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
                Isyn.Add(synVector);
            }
            surfaceArea = Vector.Build.Dense(Neuron.nodes.Count, 0.0);

            Mpre = M.Clone(); Npre = N.Clone(); Hpre = H.Clone();

            // Set color bar range directly on ColorLUT to bypass serialized field defaults
            ColorLUT.GlobalMin = -0.07f;   // -70 mV (VStart)
            ColorLUT.GlobalMax = 0.05f;   //   50 mV
        }
        /// <summary>
        /// This is for constructing the lhs and rhs of system matrix \n
        /// This will construct a HINES matrix (symmetric), it should be tridiagonal with some off
        /// diagonal entries corresponding to a branch location in the neuron graph \n
        /// The entries are defined by the following:
        /// \f[
        /// \left(-\sum_{k\in\mathcal{N}_j}\eta_kV_k^{n+1}\right)+\omega_jV_j^{n+1}=\left(\sum_{k\in\mathcal{N}_j}\eta_kV_k^{n}\right)+\bar{\omega}_jV_j^{n}
        /// \f]
        /// where
        /// \f[\eta_k = \frac{\gamma_{k, j}\Delta t}{ 2}\f]
        /// and
        /// \f[\omega_j = 1+\frac{\theta_j\Delta t}{ 2} = 1 +\frac{\Delta t\sum_{ p\in\mathcal{ N} _j}\gamma_{ p,j} }{ 2}\f]
        /// and \f$\gamma_{ k,j}\f$ is defined as
        /// \f[\gamma_{k, j}:=\frac{ 1}{ C_mR_a a_j\widetilde{\Delta x_j} }\cdot \frac{ 1}{\left(\frac{ 1} { a_{ k} ^2} +\frac{ 1} { a_j ^ 2}\right)\Delta x_{ { k},j} }\f]
        /// </summary>
        /// <param name="myCell"></param> this is the <c>Neuron</c> that contains all the information about the cell geometry
        /// <param name="res"></param> this is the axial resistance
        /// <param name="cap"></param> this is the membrane capacitance
        /// <param name="k"></param> this is the fixed time step size
        /// <returns>LHS,RHS</returns> the function returns the LHS, RHS stencil matrices for the diffusion solve in sparse format, it is compressed in the main solver routine.
        public static List<CoordinateStorage<double>> makeSparseStencils(Neuron myCell, double res, double cap, double k)
        {
            /// send output matrices as a list {rhs, lhs}\n
            /// <c>List<CoordinateStorage<double>> stencils = new List<CoordinateStorage<double>>();</c> initializes empty list storage for the stencil matrices \n
            /// in this case they are of type <c>CoordinateStorage</c> \n
            List<CoordinateStorage<double>> stencils = new List<CoordinateStorage<double>>();

            /// initialize new coordinate storage \n
            /// <c>var rhs = new CoordinateStorage<double>(myCell.vertCount, myCell.vertCount, myCell.vertCount * myCell.vertCount);</c> this initializes our empty coordinate storage matrices <c>rhs</c> and <c>lhs</c>
            var rhs = new CoordinateStorage<double>(myCell.nodes.Count, myCell.nodes.Count, myCell.nodes.Count * myCell.nodes.Count);
            var lhs = new CoordinateStorage<double>(myCell.nodes.Count, myCell.nodes.Count, myCell.nodes.Count * myCell.nodes.Count);

            /// for keeping track of the neighbors of a node \n
            /// <c>List<int> nghbrlist;</c> this is for collecting the neighbor indices of the current node 
            List<int> nghbrlist;
            int nghbrLen;

            double tempEdgeLen, tempRadius, avgEdgeLengths;
            /// <c>double sumRecip = 0;</c> this is for adding the sum of reciprocals which is in our stencil scheme \n
            double sumRecip = 0;
            double scf = 1E-6;  /// 1e-6 scale factor to convert to micrometers for radii and edge length \n

            for (int j = 0; j < myCell.nodes.Count; j++)
            {
                List<double> edgelengths = new List<double>();
                /// <c>nghbrlist = myCell.nodes[j].AdjacencyList.Keys.ToList();</c> this gets the current neighbor list for node j \n
                nghbrlist = myCell.nodes[j].AdjacencyList.Keys.ToList();
                /// <c>nghbrLen = nghbrlist.Count;</c> this is the length of the neighbor list \n
                nghbrLen = nghbrlist.Count;
                sumRecip = 0;
                /// <c>tempRadius = myCell.nodeData[j].nodeRadius*scf;</c> get the current radius at node j \n
                tempRadius = myCell.nodes[j].NodeRadius*scf;

                /// in this loop we collect the edgelengths that go to node j, and we compute the coefficient given in our paper \n
                foreach (int nghbrIds in nghbrlist)
                {
                    /// <c>tempEdgeLen = myCell.nodes[j].AdjacencyList[nghbrIds]*scf;</c> get the edge length at current node j, to node neighbor p, scale to micro meters \n
                    tempEdgeLen = myCell.nodes[j].AdjacencyList[nghbrIds]*scf;
                    /// <c>edgelengths.Add(tempEdgeLen);</c> put the edge length in the list, this list of edges will have length equal to length of neighbor list \n
                    edgelengths.Add(tempEdgeLen);
                    sumRecip = sumRecip + 1 / (tempEdgeLen * tempRadius * ((1 / (myCell.nodes[nghbrIds].NodeRadius*scf* myCell.nodes[nghbrIds].NodeRadius*scf)) + (1 / (tempRadius * tempRadius))));
                }
                /// control-volume length at node j: half the sum of its incident
                /// edge lengths (paper Sec. 3.1.1; matches
                /// NeuroVISOR-CSharpStudies/src/csharp/Sbdf2Solver.cs's
                /// HalfEdgeMembraneLength). The plain average instead gave end
                /// nodes a full edge and branch nodes sum/3, making the spatial
                /// discretization only first order.
                avgEdgeLengths = 0.5 * edgelengths.Sum();
                /// set main diagonal entries using <c>rhs.At()</c>
                /// this is BE method, no oscillations but not as accurate!
                rhs.At(j, j, 1.0);
                lhs.At(j, j, 1 + ((2.0/3.0)*k * sumRecip) / (1.0 * res * cap * avgEdgeLengths));

                /// This is for CN method, this will cause oscillations and flickering!
                //rhs.At(j, j, 1 - (k * sumRecip) / (2.0 * res * cap * avgEdgeLengths));
                //lhs.At(j, j, 1 + (k * sumRecip) / (2.0 * res * cap * avgEdgeLengths));
                /// set off diagonal entries by going through the neighbor list, and using <c>rhs.At()</c>
                for (int p = 0; p < nghbrLen; p++)
                {
                    // this is for CN method, notice the factor of 2
                    //rhs.At(j, nghbrlist[p], k / (2 * res * cap * tempRadius* avgEdgeLengths * edgelengths[p] * ((1 / (myCell.nodes[nghbrlist[p]].NodeRadius*scf * myCell.nodes[nghbrlist[p]].NodeRadius*scf)) + (1 / (tempRadius * tempRadius)))));
                    lhs.At(j, nghbrlist[p], -1.0*(2.0/3.0) * k / (1.0 * res * cap * tempRadius * avgEdgeLengths * edgelengths[p] * ((1 / (myCell.nodes[nghbrlist[p]].NodeRadius*scf * myCell.nodes[nghbrlist[p]].NodeRadius*scf)) + (1 / (tempRadius * tempRadius)))));
                }
            }
            //rhs.At(0, 0, 1);
            //lhs.At(0, 0, 1);

            /// <c>stencil.Add()</c> this adds the completed stencil matrices to the output list
            stencils.Add(rhs);
            stencils.Add(lhs);
            return stencils;
        }

        /// <summary>
        /// This is the reaction term of the HH equation which is defined by
        /// \f[ r(V):=-\frac{\bar{g}_{K}}{C}n^4(V-V_k)-\frac{\bar{g}_{Na}}{C}m^3h(V-V_{Na})-\frac{\bar{g}_l}{C}(V-V_l) \f]
        /// </summary>
        /// <param name="reactConst"></param> these are the conductances and reversal potentials defined by <c>List<double> reactConst = new List<double> { gk, gna, gl, ek, ena, el };</c>
        /// <param name="V"></param> this is the voltage vector
        /// <param name="NN"></param> this is the state vector n
        /// <param name="MM"></param> this is the state vector m
        /// <param name="HH"></param> this is the state vector h
        /// <param name="cap"></param> this is the capacitance
        /// <returns></returns>
        private static Vector reactF(List<double> reactConst, Vector V, Vector NN, Vector MM, Vector HH, double cap)
        {
            /// initialize the output vector and prod vector these will be used to assemble the different parts of the reaction calculation \n
            /// <c>Vector output = Vector.Build.Dense(V.Count, 0.0);</c> initializes the output vector of length equation to number of entries in voltage vector, initialized to 0 \n
            /// <c>Vector prod = Vector.Build.Dense(V.Count, 0.0);</c> initializes the product vector of length equation to number of entries in voltage vector, initialized to 0 \n
            /// <c>double ek, ena, el, gk, gna, gl; </c> these are the conductances and reversal potentials that we need to assign using <c>reactConst</c> parameter that is sent \n
            Vector output = Vector.Build.Dense(V.Count, 0.0);
            Vector prod = Vector.Build.Dense(V.Count, 0.0);
            double ek, ena, el, gk, gna, gl;
            /// this sets the constants for the conductances \n
            /// <c>gk = reactConst[0]; gna = reactConst[1]; gl = reactConst[2];</c>
            gk = reactConst[0]; gna = reactConst[1]; gl = reactConst[2];
            /// this sets constants for reversal potentials \n
            /// <c>ek = reactConst[3]; ena = reactConst[4]; el = reactConst[5];</c>
            ek = reactConst[3]; ena = reactConst[4]; el = reactConst[5];
            /// <c>output.Add(prod.Multiply(gk), output);</c> this adds current due to potassium
            prod.SetSubVector(0, V.Count, NN.PointwisePower(4.0));
            prod.SetSubVector(0, V.Count, (V.Subtract(ek)).PointwiseMultiply(prod));
            output.Add(prod.Multiply(gk), output);
            /// <c>output.Add(prod.Multiply(gna), output);</c> this adds current due to sodium
            prod.SetSubVector(0, V.Count, MM.PointwisePower(3.0));
            prod.SetSubVector(0, V.Count, HH.PointwiseMultiply(prod)); prod.SetSubVector(0, V.Count, (V.Subtract(ena)).PointwiseMultiply(prod));
            output.Add(prod.Multiply(gna), output);
            /// <c>output.Add((V.Subtract(el)).Multiply(gl), output);</c> this adds leak current
            output.Add((V.Subtract(el)).Multiply(gl), output);
            /// Return the negative of the total
            output.Multiply(-1.0 / cap, output);

            return output;
        }

        private void explicitUpdate(Vector S, Vector Spre, Vector F, Vector Fpre, double dt, Vector scale)
        {
            S.Add(F.PointwiseMultiply(scale.Multiply(dt)), S);
            S.Multiply(4.0 / 3.0, S);
            S.Add(Spre.Multiply(-1.0 / 3.0), S);
            S.Add(Fpre.PointwiseMultiply(scale.Multiply(-2.0 * dt / 3.0)), S);
        }
        
        private void explicitUpdate(Vector S, Vector Spre, Vector F, Vector Fpre, double dt, double scale)
        {
            S.Add(F.Multiply(dt * scale), S); 
            S.Multiply(4.0 / 3.0, S);
            S.Add(Spre.Multiply(-1.0 / 3.0), S); 
            S.Add(Fpre.Multiply(-2.0 * dt * scale / 3.0), S);
        }

        /// <summary>
        /// This is the function for the right hand side of the ODE on state S, which is given by:
        /// \f[\frac{dS}{dt}=\alpha_S(V)(1-S)-\beta_S(V)S\f]
        /// </summary>
        /// <param name="a"></param> this is the rate vector
        /// <param name="b"></param> this is the rate vector
        /// <param name="S"></param> this is the current vector of state S for the geometry
        /// <returns>f(V,N)</returns> the function returns the right hand side of the state N ODE.
        private static Vector fS(Vector S, Vector a, Vector b, double dt) 
        {
            var fac = (a + b).Map(v => ScalingFactor(v, dt));

            var aScaled = a.PointwiseMultiply(fac);
            var bScaled = b.PointwiseMultiply(fac);
            return aScaled.PointwiseMultiply(1 - S) - bScaled.PointwiseMultiply(S);
        }
        private static double ScalingFactor(double v, double dt)
        {
            if (System.Math.Abs(v) < 1.0E-10)
                return 1.0;
            return System.Math.Max(dt / v, 1.0);
        }
       
        /// <summary>
        /// This is \f$\alpha_n\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>an</returns> this function returns the rate at the given voltage
        private static Vector an(Vector V)
        {
            var V_arr = Vector.Build.DenseOfVector(V);
            V_arr.Multiply(1.0E3, V_arr);
            double Vt_scaled = 1.0E3 * Vt;

            var d_alpha = V_arr - Vt_scaled - 15.0;
            var alpha = Vector.Build.Dense(V.Count);

            for (int i = 0; i < V.Count; i++)
            {
                double da = d_alpha[i];

                /// L'hospital's rule for limit form when da is close to 0, 
                /// this is to avoid numerical instability and NaN values
                if (System.Math.Abs(da) < 1.0E-6)
                {
                    // limit form
                    alpha[i] = 1.0E3 * -0.032 /
                            ((-1.0 / 5.0) * System.Math.Exp(-da / 5.0));
                }
                else
                {
                    alpha[i] = 1.0E3 * -0.032 * da /
                            (System.Math.Exp(-da / 5.0) - 1.0);
                }
            }

            return alpha;
        }
        /// <summary>
        /// This is \f$\beta_n\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>bn</returns> this function returns the rate at the given voltage
        private static Vector bn(Vector V)
        {
            Vector Vin = Vector.Build.DenseOfVector(V);
            Vin.Multiply(1.0E3, Vin);
            return (1.0E3) * (0.5) * ((10.0 - Vin + 1.0E3 * Vt) / 40.0).PointwiseExp();
        }
        /// <summary>
        /// This is \f$\alpha_m\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>am</returns> this function returns the rate at the given voltage
        private static Vector am(Vector V)
        {
            var V_arr = Vector.Build.DenseOfVector(V);
            V_arr.Multiply(1.0E3, V_arr);
            double Vt_scaled = 1.0E3 * Vt;

            var d_alpha = V_arr - Vt_scaled - 13.0;

            var alpha = Vector.Build.Dense(V.Count);

            for (int i = 0; i < V.Count; i++)
            {
                double da = d_alpha[i];

                /// L'hospital's rule for limit form when da is close to 0, 
                /// this is to avoid numerical instability and NaN values
                if (System.Math.Abs(da) < 1.0E-6)
                {
                    // limit form
                    alpha[i] = -1.0E3 * 0.32 /
                            ((-1.0 / 4.0) * System.Math.Exp(-da / 4.0));
                }
                else
                {
                    alpha[i] = -1e3 * 0.32 * da /
                            (System.Math.Exp(-da / 4.0) - 1.0);
                }
            }

            return alpha;
        }
        /// <summary>
        /// This is \f$\beta_m\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>bm</returns> this function returns the rate at the given voltage
        private static Vector bm(Vector V)
        {
            var V_arr = Vector.Build.DenseOfVector(V);
            V_arr.Multiply(1.0E3, V_arr);
            double Vt_scaled = 1.0E3 * Vt;

            var d_beta = V_arr - Vt_scaled - 40.0;

            var beta = Vector.Build.Dense(V.Count);

            for (int i = 0; i < V.Count; i++)
            {
                double db = d_beta[i];

                /// L'hospital's rule for limit form when da is close to 0, 
                /// this is to avoid numerical instability and NaN values
                if (System.Math.Abs(db) < 1.0E-6)
                {
                    // limit form
                    beta[i] = 1.0E3 * 0.28 /
                            ((1.0 / 5.0) * System.Math.Exp(db / 5.0));
                }
                else
                {
                    beta[i] = 1.0E3 * 0.28 * db /
                            (System.Math.Exp(db / 5.0) - 1.0);
                }
            }

            return beta;
        }
        /// <summary>
        /// This is \f$\alpha_h\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>ah</returns> this function returns the rate at the given voltage
        private static Vector ah(Vector V)
        {
            Vector Vin = Vector.Build.DenseOfVector(V);
            Vin.Multiply(1.0E3, Vin);
            return (1.0E3) * (0.128) * ((17.0 - Vin + 1.0E3 * Vt) / 18.0).PointwiseExp();
        }
        /// <summary>
        /// This is \f$\beta_h\f$ rate function, the rate functions take the form of
        /// \f[
        /// \frac{a_p(V-B_p)}{\exp(\frac{V-B_p}{C_p})-D_p}
        /// \f]
        /// the constants \f$a_p,B_p,C_p,D_p\f$ are manually coded in for this version of the simulation\n
        /// TODO: come up with an implementation where the user can enter in their own parameters (Yale Neuron has this capability)
        /// </summary>
        /// <param name="V"></param> this is the input voltage
        /// <returns>bh</returns> this function returns the rate at the given voltage
        private static Vector bh(Vector V)
        {
            Vector Vin = Vector.Build.DenseOfVector(V);
            Vin.Multiply(1.0E3, Vin);
            return (1.0E3) * 4.0 / (((40.0 - Vin + 1.0E3 * Vt) / 5.0).PointwiseExp() + 1.0);
        }

        // used by save/load functions in Menu.cs
        public double[] getM() { return M.AsArray(); }
        public double[] getN() { return N.AsArray(); }
        public double[] getH() { return H.AsArray(); }

        public double[] getUpre() { return Upre.AsArray(); }
        public double[] getMpre() { return Mpre.AsArray(); }
        public double[] getNpre() { return Npre.AsArray(); }
        public double[] getHpre() { return Hpre.AsArray(); }
        public void BuildVectors(double[] u, double[] m, double[] n, double[] h,
                                    double[] upre, double[] mpre, double[] npre, double[] hpre)
        {
            lock (visualizationValuesLock) U = Vector.Build.DenseOfArray(u);
            lock (visualizationValuesLock) U_Active = U.Clone();
            Upre = Vector.Build.DenseOfArray(upre);

            M = Vector.Build.DenseOfArray(m);
            N = Vector.Build.DenseOfArray(n);
            H = Vector.Build.DenseOfArray(h);

            Mpre = Vector.Build.DenseOfArray(mpre);
            Npre = Vector.Build.DenseOfArray(npre);
            Hpre = Vector.Build.DenseOfArray(hpre);

            Isyn = new List<Vector>();
            for (int i = 0; i < 2; i++)
            {
                // Create a dense vector for each node
                var synVector = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
                Isyn.Add(synVector);
            }
            surfaceArea = Vector.Build.Dense(Neuron.nodes.Count, 0.0);
        }
    }
}