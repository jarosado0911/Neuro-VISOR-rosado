using UnityEngine;
using System;

/*
Reference:

Rothman, J.S. (2014). Modeling Synapses. In: Jaeger, D., Jung, R. (eds) 
Encyclopedia of Computational Neuroscience. Springer, New York, NY. 
https://doi.org/10.1007/978-1-4614-7320-6_240-1
*/

public class ModelGABA : ISynapseModel
{
    private string modelName;
    private double Erev;
    private double taud;
    private double g;
    private double Imax;
    private double voltageThreshold;   //Volts
    private double refireRate; // refire at 5%
    private double minRefireTime; // ms
    public ModelGABA() {
        modelName = "GABA";
        Erev = -0.080;           // (Volts) GABA_A reversal potential - matches the paper and
                                        // NeuroVISOR-CSharpStudies/src/csharp/GabaSynapse.cs (item 16; was -65mV)
        taud = 6.5e-3;            // (Seconds) decay constant, matches the paper and the studies (item 16; was 15ms)
        g = 1e-9;                 // (Siemens) positive, outward-positive convention (item 10) - matches the paper
                                        // and the studies' validated value (item 16; was -25nS, both the wrong
                                        // sign and an oversized demo magnitude)

        double Vmax = 0.1;  // (Volts)
        Imax = System.Math.Abs(g * (Vmax - Erev));

        voltageThreshold = 0.038;   //Volts
        refireRate = 3.0; // refire at 5%
        minRefireTime = 1.0e-2; // ms
    }

    /// This is the GABA Synapse function borrowed from Rothman.
    /// </summary>
    /// <param name="v"></param> this is the postsynaptic voltage
    /// <param name="t"></param> this is the current simulation time
    /// <param name="ts"></param> this is the activation time of the synapse, this is NOT the time the synapse is placed
    /// <returns></returns>
    public double getModelCurrent(double v, double t, double ts)
    {
        double current = g * System.Math.Exp(-(t - ts) / taud) * (v - Erev);
        
        return current;      
    }

    //Returns the model name
    public string getModelName()
    {
        return modelName;
    }

    public double getImax()
    {
        return Imax;
    }
    public double getTaud()
    {
        return taud;
    }
    
    //Boolean for synapse behavior, used for material of synapse
    public bool isExcitatory()
    {
        return false;
    }
    
    public bool isActive(double presynVoltage, double presynVoltagePrev, double SimulationTime, double ActivationTime)
    {
        bool updateActivation = false;

        if ((presynVoltage >= voltageThreshold) && ((presynVoltagePrev < voltageThreshold) || (SimulationTime - ActivationTime > refireRate*taud)) && (SimulationTime - ActivationTime > minRefireTime))
        {
            Debug.Log("Activation Time Updated");
            updateActivation = true;
        }

        return updateActivation;

    }
}