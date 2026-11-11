This project consists of several files and programs written with the goal of tracking Columbus' voyage from the Canary Islands to the Carribean during his first voyage.  It uses historical estimates of the earths geomagnetic field to estimate his course based on headings recorded in his journal.

FILES:
- 0098060 - Ocean surface drift data, empty directory as I never ended up needing it.  Could be obtained from NCEI archives if necessary
- CALS3K.x - Different historical gemoagnetic models
- columbus_distance_headings.txt - Data for Columbus' headings and distance travelled.  Obtained from Bartolome de las Casas' transcription in Historia de las Indias and converted to modern units (leauges for distance)
- columbus_monte_carlo.png - Example output of the Monte Carlo sim based off of CALS3K.4b uncertainties in geomagnetic estimates
- declination_uncertainty.py - Program to run the monte carlo sim
- declination.py - Plots course without accounting for uncertainty
- fielduncert3k.x - Fortran program included with CALS3K for estimating uncertainties, includes a compiled version for my hardware
- isogonic.py - Creates map showing isogonic field lines for a given geomagnetic field model and year
- isogonic.png - Example output of the isogonic field map for year 1492
- isogonic_perturbed.py - Shows isogonic field plots for four perturbed versions of the 1492 model
- isogonic_perturbed.png - Example output, isogonic plots of 4 perturbed geomagnetic models for year 1492