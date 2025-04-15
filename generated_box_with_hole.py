# Import necessary modules
import FreeCAD as App
import Part

# Create a new document
doc = App.newDocument("GeneratedModel")

# Define dimensions for the box (50x50x20 mm)
box_length = 50
box_width = 50
box_height = 20

# Create the box with the given dimensions
box = Part.makeBox(box_length, box_width, box_height)

# Define parameters for the cylindrical hole
# Diameter is 10 mm -> radius is 5 mm.
# We'll use a cylinder height of 30 mm to ensure it fully intersects the box.
hole_radius = 5
hole_height = 30

# Create the cylinder for the hole (default cylinder axis is along the Z-axis, base at (0,0,0))
cylinder = Part.makeCylinder(hole_radius, hole_height)

# Position the cylinder so that its top face coincides with the top face center of the box.
# The center of the top face of the box is at (box_length/2, box_width/2, box_height) which is (25,25,20).
# Since the cylinder height is 30 mm, to have its top at z=20, translate it by (25,25, -10).
cylinder.translate(App.Vector(box_length/2, box_width/2, -10))

# Subtract the cylinder from the box to create the through hole
box_with_hole = box.cut(cylinder)

# Add the resultant shape to the document
doc.addObject("Part::Feature", "BoxWithThroughHole").Shape = box_with_hole

# Recompute the document to update the view
doc.recompute()

# Optional: If running in the FreeCAD GUI, you can fit the view to the object
# try:
#     import FreeCADGui as Gui
#     Gui.SendMsgToActiveView("ViewFit")
# except ImportError:
#     pass