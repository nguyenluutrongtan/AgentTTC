FeatureScript 2625;
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Single Sphere Radius 50mm", "Feature Type Description" : "Generates a sphere with a fixed radius of 50mm at the origin." }
export const singleSphereRadius50mm = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // No user inputs; sphere radius and center are fixed by design requirement
    }
    {
        // Create a sphere of radius 50 mm centered at the origin
        opSphere(context, id, {
            "radius" : 50 * millimeter,
            "center" : vector(0, 0, 0) * millimeter
        });
    });