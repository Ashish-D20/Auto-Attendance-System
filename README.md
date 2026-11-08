# Auto-Class-Attendance
A system that detects faces in an input image and matches it against the faces in database and detects whether a face/person from the database is present in the image or not.

In this repository, names like "no_expansion", "6.25%_expansion", etc. refer to the expansion percentage applied to the bounding boxes of detected faces. This is done as sometimes bounding boxes sometimes clip out some important features that might be used in matching images. Usage of expansion clearly helps as inferred from classification reports in [![Performance Comparison](./accuracy%20reports/Performance%20Comparison.png)](./accuracy%20reports/Performance%20Comparison.png)
