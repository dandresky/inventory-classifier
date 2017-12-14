# Inventory Classifier

As an electrical engineer I have developed numerous electronic devices and partnered with operations on the introduction and ramp up of my products. From domestic fulfillment centers to large contract manufacturing facilities in Asia, there are consistent pain points and inefficiencies with regards to material handling and inventory management. Some common issues include:

- Selecting the right component
- Warehouse space utilization
- Just in time throughput
- Inefficient distribution of materials
- Lack of automation

With this project, I intend to design a recurrent neural network capable of identifying and counting inventory items in a warehouse from images of their storage locations.

### Data Used

Amazon has an automated inventory management system where items are stored in random locations based solely on space available. As part of their open data program, they have made approximately half a million images of inventory bins from one of their fulfillment centers available for research, along with JSON documents describing the contents of each of the bins that can be used to train the model on.

### Minimum Viable Product

Due to the time frames for the capstone project, the MVP will be a RNN that can count each of the distict items in a bin. As time permits, I will expand the project to classify each item and ultimately identify with unique part numbers/descriptions.
