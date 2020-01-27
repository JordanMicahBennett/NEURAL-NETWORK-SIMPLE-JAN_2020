import java.util.ArrayList;

public class Topology extends ArrayList <Integer>
{
    //features
    private String description = "";
    
    public Topology ( String description )
    {
        generateTopology ( description );
    }
    
    public void generateTopology ( String description )
    {
        String [ ] parts = description.split ( "," );
        
        for ( int pI = 0; pI < parts.length; pI ++ )
            add ( Integer.parseInt ( parts [ pI ] ) );
    }
}