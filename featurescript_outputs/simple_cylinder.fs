// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { 
    "Feature Type Name" : "Simple Cylinder",
    "Feature Type Description" : "Cylinder with radius 1 inch and height 3 inches, base on the XY‑plane at the origin"
}
export const simpleCylinder = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // This feature uses fixed dimensions—no user inputs required
    }
    {
        // Create a cylinder primitive: bottom at (0,0,0), top at (0,0,3in), radius 1in
        fCylinder(context, id + "cylinder1", {
            "bottomCenter" : vector(0, 0, 0) * inch,
            "topCenter"    : vector(0, 0, 3) * inch,
            "radius"       : 1 * inch
        });
    });