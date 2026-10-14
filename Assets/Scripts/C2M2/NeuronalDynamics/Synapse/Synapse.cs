using C2M2;
using C2M2.NeuronalDynamics.Simulation;
using C2M2.NeuronalDynamics.UGX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using UnityEngine;
public class Synapse : NDInteractables
{
    //Defines a List to be used as the list of available synaptic models
    static ISynapseModel[] modelArray = {new ModelNMDA(), new ModelGABA(), new ModelAMPA()};
    static LinkedList<ISynapseModel> modelList = new LinkedList<ISynapseModel>(modelArray);
    public LinkedListNode<ISynapseModel> currentModel = modelList.First;

    //Defines the material for the model
    public Material inhibitoryMat;
    public Material excitatoryMat;
    public Material prePlaceMat;
    public Material NMDAMat;
    public Material AMPAMat;
    public Material GABAMat;
    private static int nextId = 0;
    public int Id;
    private bool isBeingDestroyed = false;


    // Activation time t_s of this (presynaptic) synapse. It is written by the solve thread
    // of the presynaptic cell, which detects the threshold crossing, and read by the solve
    // thread of the postsynaptic cell, which can be another thread up to one step ahead or
    // behind (the solve barrier keeps all cells within one step of each other).
    private object activationLock = new object();
    private double activationTime = double.NegativeInfinity;
    private double previousActivationTime = double.NegativeInfinity;
    private int activationStep = int.MinValue;

    /// <summary>
    /// The latest activation time [s]. Setting it (placement, loading) makes it visible to
    /// every time step at once.
    /// </summary>
    public double ActivationTime
    {
        get { lock (activationLock) return activationTime; }
        set
        {
            lock (activationLock)
            {
                activationTime = value;
                previousActivationTime = value;
                activationStep = int.MinValue;
            }
        }
    }

    /// <summary>
    /// Records an activation detected at time level <paramref name="step"/>, t_s = t_step.
    /// </summary>
    public void RecordActivation(double ts, int step)
    {
        lock (activationLock)
        {
            previousActivationTime = activationTime;
            activationTime = ts;
            activationStep = step;
        }
    }

    /// <summary>
    /// The activation time to use for the synaptic current at time level <paramref name="step"/>.
    /// An activation detected at a later level (by a presynaptic cell whose thread is already one
    /// step ahead) is not visible yet, so the current does not depend on thread timing.
    /// </summary>
    public double ActivationTimeAt(int step)
    {
        lock (activationLock) return activationStep <= step ? activationTime : previousActivationTime;
    }

    public List<double> currentIsyn = new List<double>();


    //public double ts { get; set; } = 0;

    public Neuron.NodeData NodeData
    {
        get
        {
            if (simulation == null || simulation.Neuron == null || simulation.Neuron.nodes == null)
            {
                Debug.LogWarning("Synapse.NodeData: simulation, Neuron, or nodes is null");
                return null;
            }
            if (FocusVert < 0 || FocusVert >= simulation.Neuron.nodes.Count)
            {
                Debug.LogWarning("Synapse.NodeData: FocusVert " + FocusVert + " out of bounds (0.." + (simulation.Neuron.nodes.Count - 1) + ")");
                return null;
            }
            return simulation.Neuron.nodes[FocusVert];
        }
    }

    public SynapseManager SynapseManager
    {
        get
        {
            if (GameManager.instance == null || GameManager.instance.simulationManager == null)
                return null;
            return GameManager.instance.simulationManager.synapseManager;
        }
    }

    private void OnDestroy()
    {
        if (GameManager.isQuitting) return;
        if (isBeingDestroyed) return;
        isBeingDestroyed = true;
        var manager = SynapseManager;
        if (manager == null) return;
        manager.DeleteSyn(manager.FindSelectedSyn(this));
    }
    
    // Creates a unique synapse instance
    public Synapse Clone()
    {
        Synapse other = (Synapse) this.MemberwiseClone();
        other.Id = Interlocked.Increment(ref nextId);
        // Re-resolve currentModel to avoid sharing the LinkedListNode reference
        other.currentModel = modelList.Find(currentModel.Value);
        // and give the clone its own activation lock
        other.activationLock = new object();
        return other;
    }
    public override void Place(int index)
    {
        transform.localPosition = FocusPos;
        float currentVisualizationScale = (float)simulation.VisualInflation;
        float radiusScalingValue = 3f * (float)NodeData.NodeRadius;
        float heightScalingValue = 1f * simulation.AverageDendriteRadius;
        float radiusLength = Math.Max(radiusScalingValue, heightScalingValue) * currentVisualizationScale;
        transform.localScale = new Vector3(radiusLength, radiusLength, radiusLength);
        SetToModeMaterial();
    }

    protected override void AddHitEventListeners()
    {
        HitEvent.OnHoldPress.AddListener((hit) => MonitorInput());
        HitEvent.OnEndPress.AddListener((hit) => CheckInput());
    }

    public void MonitorInput()
    {
        SynapseManager.HoldCount += Time.deltaTime;
        // If we've held the button long enough to destroy, color caps red until user releases button
        if (SynapseManager.HoldCount > SynapseManager.DestroyCount) SwitchMaterial(destroyMaterial);
    }
    
    private void CheckInput()
    {
        // Change model 
        if (SynapseManager.HoldCount >= SynapseManager.ChangeCount && SynapseManager.HoldCount <= SynapseManager.DestroyCount)
        {
            //Implements the circularly linked list 
            if (currentModel.Value.Equals(modelList.Last.Value)) {
                SynapseManager.ChangeModel(SynapseManager.FindSelectedSyn(this), modelList.First.Value);
                currentModel = modelList.First;     //Resets to the first model in the linked list
            }
            else {
                SynapseManager.ChangeModel(SynapseManager.FindSelectedSyn(this), currentModel.Next.Value);
                currentModel = currentModel.Next;   //Iterates through the linked list
            }
            Debug.Log("Current Model: " + currentModel.Value);
        }
        // Delete synapse
        else if (SynapseManager.HoldCount >= SynapseManager.DestroyCount)
        {
            SynapseManager.DeleteSyn(SynapseManager.FindSelectedSyn(this));        }
        // Place synapse 
        else if (GameManager.instance.simulationManager.FeatState == NDSimulationManager.FeatureState.Synapse)
        {
            SynapseManager.SynapticPlacement(this);
        }
        SynapseManager.HoldCount = 0;
    }

    public void SwitchModel(ISynapseModel model)
    {
        currentModel = modelList.Find(model);
        SetToModeMaterial();
    }

    public void SetPrePlace()
    {
        SwitchMaterial(prePlaceMat);
    }

    public void SetToModeMaterial()
    {
        if (currentModel.Value.isExcitatory()) { SwitchMaterial(excitatoryMat); }
        else { SwitchMaterial(inhibitoryMat); }
    }
}