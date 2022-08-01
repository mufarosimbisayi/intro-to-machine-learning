#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np
    consolidated = np.stack((ages.flatten(), net_worths.flatten(), predictions.flatten()), axis=1)
    difference = consolidated[:,1] - consolidated[:,2]
    difference = difference.reshape(90,1)
    consolidated = np.hstack((consolidated,difference))
    consolidated = consolidated[consolidated[:, 3].argsort()]
    consolidated = consolidated[9:]
    consolidated = consolidated[:,[0,1,3]]
    cleaned_data = [tuple(row) for row in consolidated]
    #print(cleaned_data)
    
    return cleaned_data

