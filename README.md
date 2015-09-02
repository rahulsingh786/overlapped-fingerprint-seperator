# overlapped-fingerprint-seperator

Overview
Fingerprints are commonly used for authentication and suspect identification at crime scenes. They are one of the most valuable pieces of information in criminal investigation. However, there are current limitations. It is common to find overlapping fingerprints. Our project aims to implement a system that receives an image of overlapping fingerprints and uniquely maps each fingerprint to a given database. 

Goals
This project will be successful if both of the following conditions are met:
Develop a novel fingerprint separation algorithm (optionally including a novel matching component) that is demonstrably successful in separating multiple superimposed fingerprint images.
Implement a software application that demonstrates and visualizes the operation of this novel algorithm. The result must be interactive in a way that is informative, engaging, and intuitive to operate for a person with no knowledge of the subject domain and no specialized technology skills -- i.e. a layperson.

Specifications
Algorithm parameters: novel fingerprint algorithm must be demonstrably successful on images that contain three or more superimposed fingerprints.
Algorithm functionality: the novel algorithm must necessarily separate multiple fingerprints; it may optionally also perform selection and matching on the individual fingerprints contained in a superimposed image.
Real-world data: the algorithm must operate successfully on data that realistically represents real-world scenarios.
Application graphical design: the demo application must incorporate a graphical user interface that utilizes common cues and workflows.
Application use case: the demo application will
Allow a user to generate their own superimposed image out of a selection of stock single print images. This feature should allow for some custom user manipulation, such as print rotation angle or translation.
Use the actual separation algorithm in real time to attempt separation on the user-generated input.
Upon algorithm termination, the application will display the results of the attempted separation. Additionally, it should display relevant visualizations of internal logic that will help inform the user of the algorithms’ operation, provided it is feasible to generate such images.
Application functionality: the application will demonstrate both fingerprint separation and matching/retrieval. In the case that no novel fingerprint matching component is included, a domain-standard matching algorithm may be used to perform this task.
