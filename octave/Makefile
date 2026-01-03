# Generate electromagnetic field plots using Octave

OCTAVE = octave --no-gui --eval

all: bfield_octave.png efield_octave.png

bfield_octave.png: plot_bfield.m loop_bfield.m
	$(OCTAVE) "plot_bfield(1.0, 1.0, 2.0, 'bfield_octave.png')"

efield_octave.png: plot_efield.m loop_efield.m
	$(OCTAVE) "plot_efield(1.0, 1e-9, 2.0, 'efield_octave.png')"

clean:
	rm -f bfield_octave.png efield_octave.png

.PHONY: all clean
