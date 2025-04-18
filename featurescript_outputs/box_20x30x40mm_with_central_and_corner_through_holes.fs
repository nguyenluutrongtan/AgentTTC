// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/geometry.fs", version : "2625.0");
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Box 20x30x40mm with central and corner through holes" }
export const box20x30x40WithCentralCornerHoles = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // No user inputs required
    }
    {
        // Create the base box of 20×30×40 mm
        fCuboid(context, id + "baseBox", {
            "corner1" : vector(0, 0, 0) * mm,
            "corner2" : vector(20, 30, 40) * mm
        });

        // Create a central hole cylinder (radius 10 mm, full height)
        fCylinder(context, id + "centralHole", {
            "bottomCenter" : vector(10, 15, 0) * mm,
            "topCenter"    : vector(10, 15, 40) * mm,
            "radius"       : 10 * mm
        });

        // Create four corner hole cylinders (radius 2 mm, full height)
        fCylinder(context, id + "cornerHole1", {
            "bottomCenter" : vector(2,  2,  0) * mm,
            "topCenter"    : vector(2,  2,  40) * mm,
            "radius"       : 2 * mm
        });
        fCylinder(context, id + "cornerHole2", {
            "bottomCenter" : vector(18, 2,  0) * mm,
            "topCenter"    : vector(18, 2,  40) * mm,
            "radius"       : 2 * mm
        });
        fCylinder(context, id + "cornerHole3", {
            "bottomCenter" : vector(18, 28, 0) * mm,
            "topCenter"    : vector(18, 28, 40) * mm,
            "radius"       : 2 * mm
        });
        fCylinder(context, id + "cornerHole4", {
            "bottomCenter" : vector(2,  28, 0) * mm,
            "topCenter"    : vector(2,  28, 40) * mm,
            "radius"       : 2 * mm
        });

        // Subtract the central hole from the base box
        opBoolean(context, id + "cutCenter", {
            "tools"         : qCreatedBy(id + "centralHole", EntityType.BODY),
            "targets"       : qCreatedBy(id + "baseBox",      EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        });

        // Subtract each corner hole sequentially
        opBoolean(context, id + "cutCorner1", {
            "tools"         : qCreatedBy(id + "cornerHole1", EntityType.BODY),
            "targets"       : qCreatedBy(id + "baseBox",      EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        });
        opBoolean(context, id + "cutCorner2", {
            "tools"         : qCreatedBy(id + "cornerHole2", EntityType.BODY),
            "targets"       : qCreatedBy(id + "baseBox",      EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        });
        opBoolean(context, id + "cutCorner3", {
            "tools"         : qCreatedBy(id + "cornerHole3", EntityType.BODY),
            "targets"       : qCreatedBy(id + "baseBox",      EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        });
        opBoolean(context, id + "cutCorner4", {
            "tools"         : qCreatedBy(id + "cornerHole4", EntityType.BODY),
            "targets"       : qCreatedBy(id + "baseBox",      EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        });
    });