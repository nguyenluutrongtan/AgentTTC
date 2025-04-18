// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/geometry.fs", version : "2625.0");

annotation { "Feature Type Name" : "10×10×10 Box" }
export const box10x10x10 = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // No parameters required for a fixed-size box
    }
    {
        // Create a cuboid from the origin to (10, 10, 10) millimeters
        fCuboid(context, id + "cuboid1", {
            "corner1" : vector(0, 0, 0) * millimeter,
            "corner2" : vector(10, 10, 10) * millimeter
        });
    });