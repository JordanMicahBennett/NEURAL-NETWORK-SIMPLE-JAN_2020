public class NeuralNetwork
{
    //features
    private double eta = 0.2;
    private double alpha = 0.5;
    private Topology topology = new Topology ( "2,2,1" );
    private Layers layers = new Layers ( );
    
    public NeuralNetwork ( )
    {
        for ( int tI = 0; tI < topology.size ( ); tI ++ )
        {
            layers.add ( new Layer ( ) ); 
            
            for ( int lSI = 0; lSI <= topology.get ( tI ); lSI ++ )
            {
                int numberOfWeightsFromNextLayer = tI + 1 < topology.size ( ) ? topology.get ( tI + 1 ): 0;
               
                layers.get ( tI ).add ( new Neuron ( lSI, eta, alpha, numberOfWeightsFromNextLayer ) );
            }
            
            layers.get ( tI ).get ( layers.get ( tI ).size ( ) - 1 ).setOutcome ( 1.0 ); //set bias neuron
        }
    }
    
    public void doForwardPropagation ( int [ ] inputs )
    {
        //set first layer item
        for ( int iI = 0; iI < inputs.length; iI ++ )
            layers.get ( 0 ).get ( iI ).setOutcome ( inputs [ iI ] );
        
        //do forward prop based on above, starting from non-zero layer
        for ( int tI = 1; tI < topology.size ( ); tI ++ )
        {
            Layer priorLayer = layers.get ( tI - 1 );
            
            for ( int lI = 0; lI < topology.get ( tI ); lI ++ )
                layers.get ( tI ).get ( lI ).doForwardPropagation ( priorLayer );
        }
    }
    
    public void doBackwardPropagation ( int target ) 
    {
        Neuron firstNeuronInLastLayer = layers.get ( layers.size ( ) - 1 ).get ( 0 );
        
        //outcome gradient calculation
        firstNeuronInLastLayer.calculateOutcomeGradient ( target );
        
        //hidden gradient calculation
        for ( int tI = topology.size()-2; tI > 0; tI -- )
        {
            Layer currentLayer = layers.get ( tI );
            Layer nextLayer = layers.get ( tI + 1 );
            
            for ( int lSI = 0; lSI < currentLayer.size ( ) - 1; lSI ++ )
                currentLayer.get ( lSI ).calculateHiddenGradient ( nextLayer );
        }
        
        //weight update
        for ( int tI = topology.size()-1; tI > 0; tI -- )
        {
            Layer currentLayer = layers.get ( tI );
            Layer priorLayer = layers.get ( tI - 1 );
            
            for ( int lSI = 0; lSI < currentLayer.size ( ) - 1; lSI ++ )
                currentLayer.get ( lSI ).updateWeights ( priorLayer );
        }
    }
    
    public double getOutcome ( )
    {
        return layers.get ( layers.size ( ) - 1 ).get ( 0 ).getOutcome ( );
    }
}