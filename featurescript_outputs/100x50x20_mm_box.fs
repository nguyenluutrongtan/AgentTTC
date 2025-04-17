// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/geometry.fs", version : "2625.0");

// Creates a fixed 100×50×20 mm rectangular prism
annotation { "Feature Type Name" : "100x50x20 mm Box" }
export const box100x50x20mm = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // No user parameters needed for a fixed-dimension box
    }
    {
        // Construct a cuboid from the origin to (100,50,20) mm
        fCuboid(context, id + "cuboid1", {
                "corner1" : vector(0, 0, 0) * mm,
                "corner2" : vector(100, 50, 20) * mm
        });
    });