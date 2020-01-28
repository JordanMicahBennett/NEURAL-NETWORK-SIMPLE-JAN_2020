import java.util.ArrayList;
import java.util.Random;

public class Neuron
{
    //features
    private int neuronId;
    private int numberOfWeightsFromNextNeuron;
    private double eta;
    private double alpha;
    private double outcome;
    private double gradient;
    private ArrayList <Synapse> weights = new ArrayList <Synapse> ( );
    
    //constructor
    public Neuron ( int neuronId, double eta, double alpha, int numberOfWeightsFromNextNeuron )
    {
        this.neuronId = neuronId;
        this.eta = eta;
        this.alpha = alpha;
        this.numberOfWeightsFromNextNeuron = numberOfWeightsFromNextNeuron;
        gradient = 0;
        
        //initialize weights
        for ( int wI = 0; wI < numberOfWeightsFromNextNeuron; wI ++ )
        {
            weights.add ( new Synapse ( ) );
            weights.get ( wI ).setWeight ( new Random ( ).nextDouble ( ) );
        }
    }
    
    //getters
    public double getOutcome ( )
    {
        return outcome;
    }
    
    public double getGradient ( )
    {
        return gradient;
    }
    
    public ArrayList <Synapse> getWeights ( )
    {
        return weights;
    }
    
    public double getActivation ( double value )
    {
        return Math.tanh ( value );
    }
    
    public double getPrimeActivation ( double value )
    {
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }
    
    public double getDistributedWeightSigma ( Layer nextLayer )
    {
        double sigma = 0;
        
        for ( int nLI = 0; nLI < nextLayer.size ( ) - 1; nLI ++ )
            sigma += getWeights ( ).get ( nLI ).getWeight ( ) * nextLayer.get ( nLI ).getGradient ( );
            
        return sigma;
    }
    
    //setters
    public void setGradient ( double value )
    {
        gradient = value;
    }
    
    public void setOutcome ( double value )
    {
        outcome = value;
    }
    
    public void doForwardPropagation ( Layer priorLayer )
    {
        double sigma = 0;
        
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
            sigma += priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) * priorLayer.get ( pLI ).getOutcome ( );
            
        setOutcome ( getActivation ( sigma ) );
    }
    
    public void calculateHiddenGradient ( Layer nextLayer )
    {
        double delta = getDistributedWeightSigma ( nextLayer );
        
        setGradient ( getPrimeActivation ( outcome ) * delta );
    }
    
    public void calculateOutcomeGradient ( int target )
    {
        double delta = target - outcome;
        
        setGradient ( getPrimeActivation ( outcome ) * delta );
    }
    
    public void updateWeights ( Layer priorLayer )
    {
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
        {
            double oldDeltaWeight = priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getDeltaWeight ( );
            
            //newDeltaWeight = ( eta * gradient * priorOutcome ) + ( alpha * oldDeltaWeight );
            double newDeltaWeight = ( eta * getGradient ( ) * priorLayer.get ( pLI ).getOutcome ( ) ) + ( alpha * oldDeltaWeight );
            
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setDeltaWeight ( newDeltaWeight );
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setWeight ( priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) + newDeltaWeight );
        }
    }
}