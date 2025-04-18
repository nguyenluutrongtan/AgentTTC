// FeatureScript code starts here
FeatureScript 2625;
import(path : "onshape/std/geometry.fs", version : "2625.0");
import(path : "onshape/std/common.fs", version : "2625.0");

annotation { "Feature Type Name" : "Box with Tangent Top Sphere" }
export const boxWithTangentTopSphere = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        // Box dimensions
        annotation { "Name": "Box Length", "Default": 100 * millimeter }
        isLength(definition.boxLength, LENGTH_BOUNDS);

        annotation { "Name": "Box Width", "Default": 60 * millimeter }
        isLength(definition.boxWidth, LENGTH_BOUNDS);

        annotation { "Name": "Box Height", "Default": 40 * millimeter }
        isLength(definition.boxHeight, LENGTH_BOUNDS);

        // Sphere radius
        annotation { "Name": "Sphere Radius", "Default": 25 * millimeter }
        isLength(definition.sphereRadius, LENGTH_BOUNDS);
    }
    {
        // Create the base box from the world origin to the specified dimensions
        var corner1 = vector(0, 0, 0) * definition.boxLength; // yields a zero-length vector of correct type
        var corner2 = vector(
            definition.boxLength,
            definition.boxWidth,
            definition.boxHeight
        );
        fCuboid(context, id + "baseBox", {
                "corner1": corner1,
                "corner2": corner2
        });

        // Compute the sphere center: midpoint of the top face, offset by the radius
        var sphereCenter = vector(
            definition.boxLength / 2,
            definition.boxWidth  / 2,
            definition.boxHeight + definition.sphereRadius
        );
        opSphere(context, id + "topSphere", {
                "center": sphereCenter,
                "radius": definition.sphereRadius
        });

        // Fuse (union) the box and the sphere into a single body
        opBoolean(context, id + "boxSphereUnion", {
                "targets"      : qCreatedBy(id + "baseBox",  BodyType.SOLID),
                "tools"        : qCreatedBy(id + "topSphere", BodyType.SOLID),
                "operationType": BooleanOperationType.UNION
        });
    });