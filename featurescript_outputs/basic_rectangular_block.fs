// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Basic Rectangular Block" }
export const basicRectangularBlock = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // X‐direction length
        annotation { "Name" : "Length" }
        definition.length is Length;

        // Y‐direction width
        annotation { "Name" : "Width" }
        definition.width is Length;

        // Z‐direction height (extrusion depth)
        annotation { "Name" : "Height" }
        definition.height is Length;
    }
    {
        /***************************************************************************
        * IMPLEMENTATION:
        * 1) Define the XY plane at the model origin.
        * 2) Sketch a rectangle from (0,0) to (Length,Width).
        * 3) Extrude the rectangle by Height to form the block.
        ***************************************************************************/

        // 1) Build the XY sketch plane at the origin
        var origin     = vector(0, 0, 0) * millimeter;
        var normalZ    = vector(0, 0, 1);
        var sketchPlane = skPlane(context, { "plane" : plane(origin, normalZ) });

        // 2) Create the rectangle sketch
        var sketch = opSketch(context, "rectSketch", { "sketchPlane" : sketchPlane });
        var corner1 = vector(0, 0) * millimeter;
        // Convert length and width to numeric then back to a vector with units
        var corner2 = vector(definition.length / millimeter, definition.width / millimeter) * millimeter;
        skRectangle(context, "rectangle", {
            "sketch"  : sketch,
            "corner1" : corner1,
            "corner2" : corner2
        });

        // 3) Extrude the sketch region upward by 'Height'
        opExtrude(context, "extrudeBlock", {
            "entities" : qSketchRegion(sketch),
            "direction": evOwnerSketchPlane(context, { "entity" : sketch }).normal,
            "endBound" : BoundingType.BLIND,
            "endDepth" : definition.height
        });
    }
);