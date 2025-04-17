// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Rectangular Box 20×30×40 mm with Through Hole Radius 10 mm" }
export const rectangularBox20x30x40Hole10 = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        // Sketch the 20×30 mm rectangle on the Top plane
        var skBox = newSketchOnPlane(context, id + "skBox", {
                "sketchPlane" : qCreatedBy(makeId("Top"), EntityType.FACE)
        });
        skRectangle(skBox, "rect1", {
                "firstCorner" : vector(0, 0) * millimeter,
                "secondCorner" : vector(20, 30) * millimeter
        });
        skSolve(skBox);

        // Extrude the rectangle to create the box (20×30×40 mm)
        extrude(context, id + "boxExtrude", {
                "entities" : qSketchRegion(id + "skBox"),
                "endBound" : BoundingType.BLIND,
                "depth" : 40 * millimeter
        });

        // Sketch the circle (radius 10 mm) at (10 mm, 15 mm) on the Top plane
        var skHole = newSketchOnPlane(context, id + "skHole", {
                "sketchPlane" : qCreatedBy(makeId("Top"), EntityType.FACE)
        });
        skCircle(skHole, "circle1", {
                "center" : vector(10, 15) * millimeter,
                "radius" : 10 * millimeter
        });
        skSolve(skHole);

        // Extrude the circle through all to create the cylindrical tool
        extrude(context, id + "holeExtrude", {
                "entities" : qSketchRegion(id + "skHole"),
                "endBound" : BoundingType.THROUGH_ALL
        });

        // Subtract the cylinder from the box to form the through hole
        opBoolean(context, id + "cutHole", {
                "tools" : qCreatedBy(id + "holeExtrude", EntityType.BODY),
                "targets" : qCreatedBy(id + "boxExtrude", EntityType.BODY),
                "operationType" : BooleanOperationType.SUBTRACTION
        });
    });