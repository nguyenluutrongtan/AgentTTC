// Basic Cube FeatureStudio
FeatureScript 2455;
import(path : "onshape/std/common.fs", version : "2455.0");

annotation { "Feature Type Name" : "Basic Cube" }
export const basicCube = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Cube Size", "Default" : 100 * millimeter }
        isLength(definition.size, LENGTH_BOUNDS);
    }
    {
        // Create a sketch on the Top plane for the base square
        var sketch1 = newSketch(context, id + "sketch1", {
                "sketchPlane" : qCreatedBy(makeId("Top"), EntityType.FACE)
        });
        // Draw a square with side length equal to Cube Size
        skRectangle(sketch1, "rect1", {
                "firstCorner" : vector(0, 0),
                "secondCorner" : vector(definition.size, definition.size)
        });
        skSolve(sketch1);
        // Extrude the square into a cube
        opExtrude(context, id + "extrude1", {
                "entities"  : qSketchRegion(id + "sketch1"),
                "direction" : evOwnerSketchPlane(context, { "entity" : qSketchRegion(id + "sketch1") }).normal,
                "endBound"  : BoundingType.BLIND,
                "endDepth"  : definition.size
        });
    });