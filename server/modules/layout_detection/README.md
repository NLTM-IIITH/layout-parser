# Layout Parser API

## Overview

Implementation of the Layout Parser API, specifically intergration of the Layout Detection code for various classes (Tables, Equations, Figures and Text) from Bhasini OCR API Repository.

## Changes Made
### layout_detection module
- Introduced layout_detection module, consolidating layout detection code for Tables, Equations, Figures, and Text, aiming to enhance code organization and maintainability in the Layout Parser API.
### routes.py
- Introduced `/detect-layouts` as the primary endpoint for performing layout detection across different classes in the API.
- User inputs an images and a json response containing the bounding boxes for various classes is returned.
### helpers.py
- `helpers.py` provides a simple way to perform layout detection on single images.
- The `get_layout_from_single_image` function utilizes functions from various classes (tables.py, equations.py, figures.py) to extract bounding boxes for tables, cells, equations, and figures.
- The results are organized into a dictionary, including details about the user input image as well.
### classes directory
- Set up a handy classes folder with essential files like tables.py,equations.py,figures.py
- These act as simple helpers to kickstart the layout detection process for various elements in the Layout Parser API.

## Input and Example JSON Response

**Input:**
- An image file.

**Example JSON Response for Success:**
```json
{
    "message": "Layout Detection Successful",
    "layout": {
        "image-name": "BCC_BR_115_373 (1).jpg",
        "height": 1754,
        "width": 1240,
        "tables": [
            [178, 920, 1071, 1547],
            [204, 75, 503, 204]
            // ... additional bounding boxes for tables
        ],
        "cells": [
            [197, 1425, 380, 1503],
            [198, 1018, 383, 1099]
            // ... additional bounding boxes for cells
        ],
        "equations": [
            // ... bounding boxes for equations
        ],
        "figures": [
            // ... bounding boxes for figures
        ]
    }
}