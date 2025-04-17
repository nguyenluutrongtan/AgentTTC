FeatureScript 2625;
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Rectangular Prism" }
export const rectangularPrism = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Length", "Default" : 100 * millimeter }
        isLength(definition.length, LENGTH_BOUNDS);

        annotation { "Name" : "Width", "Default" : 50 * millimeter }
        isLength(definition.width, LENGTH_BOUNDS);

        annotation { "Name" : "Height", "Default" : 25 * millimeter }
        isLength(definition.height, LENGTH_BOUNDS);
    }
    {
        // Create a sketch on the Top plane
        var sketch1 = newSketch(context, id + "sketch1", {
            "sketchPlane" : qCreatedBy(makeId("Top"), EntityType.FACE)
        });

        // Draw the base rectangle at the sketch origin
        skRectangle(sketch1, "baseRect", {
            "firstCorner" : vector(0, 0),
            "secondCorner" : vector(definition.length, definition.width)
        });
        skSolve(sketch1);

        // Extrude the sketch region to form the rectangular prism
        opExtrude(context, id + "extrude1", {
            "entities"  : qSketchRegion(id + "sketch1"),
            "direction" : evOwnerSketchPlane(context, { "entity" : qSketchRegion(id + "sketch1") }).normal,
            "endBound"  : BoundingType.BLIND,
            "endDepth"  : definition.height
        });
    });