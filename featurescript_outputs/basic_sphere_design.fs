// FeatureScript code starts here
import(path : "onshape/std/commonImports.fs", version : "1548");

// This custom feature creates a basic sphere by revolving a half‐circle profile about its diameter.
annotation { "Feature Type Name" : "Basic Sphere" }
export const sphereFeature = feature(function(id, definition)
{
    // Precondition Block: Define all required parameters for the sphere.
    precondition
    {
        // The sphere’s radius. Default is 10 mm.
        annotation { "Name" : "Radius" }
        definition.radius is Length;
    }
    // Return parameter information (can be used for UI defaults or later queries)
    return {
        "radius" : definition.radius
    };
})
-> function(context is Context, definition is map)
{
    // Main Feature Block: Create a sphere by revolving a half circle.

    // 1. Create a new sketch on the Top plane (XY plane).
    var sketchPlane = qNamedPlane(context, "Top");
    var sketch = newSketch(context, id + "sketch", { "sketchPlane" : sketchPlane });

    // 2. Define the sphere radius.
    var r = definition.radius;  // radius (e.g., 10 mm)

    // 3. Draw a half circle arc.
    //    The arc is drawn with its center at the origin. It starts at (-r, 0) and ends at (r, 0)
    //    so that the arc represents the upper half of a full circle.
    var arc = skArc(sketch, {
        "center" : vector(0, 0),
        "start" : vector(-r, 0),
        "end" : vector(r, 0)
    });

    // 4. Draw a straight line (the chord) connecting the endpoints of the arc.
    //    This closes the profile so that we can use it in the revolve operation.
    var line = skLine(sketch, {
        "start" : vector(r, 0),
        "end" : vector(-r, 0)
    });

    // 5. Solve the sketch to register the geometry.
    skSolve(sketch);

    // 6. Retrieve the complete, closed sketch region to be used as the profile.
    var profile = qSketchRegion(sketch);

    // 7. Define the axis of revolution.
    //    The axis is defined using the line (chord) we just created.
    var axis = qSketchSegment(sketch, line);

    // 8. Revolve the profile 360° about the defined axis to create the spherical body.
    var revolveOp = opRevolve(context, id + "revolve", {
        "entities" : profile,
        "axis" : axis,
        "angle" : 360 * degree
    });

    // 9. Return the created body (or bodies) as the feature output.
    return {
        "sphere" : revolveOp.bodies
    };
};