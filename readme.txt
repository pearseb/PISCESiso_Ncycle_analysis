These files contain python and ferret scripts used in the analyis for the publication:
	
	Buchanan PJ, Aumont O, Bopp L, Mahaffey C, and Tagliabue A (2021): 
	"Impact of intensifying nitrogen limitation of ocean net primary 
         production is fingerprinted by nitrogen isotopes". Nature Communications.


All scripts provided here are designed to be run using the model output, a link to which is provided in the paper.
I additionally assume that the user has the following software:
	- Ferret
	- Python, with the following packages
		- numpy
		- netCDF4
		- matplotlib
		- scipy
		- seaborn
		- cmocean
		- basemap
		- cartopy
		- tqdm
	- NCO (netcdf operators)
	- bash scripting (for "make_toe_curves.sh")



Python scripts are named according to the figures produced (main# or supp#), or according to the analysis done:
	- process-d15Nno3_observations_on_model_grid.py	: puts observations onto model grid
	- process-model_assessment.py			: computes visual and univariate statistical model-data assessment
							  also saves supplementary figure 1
	- process-compute_toe.py			: computes ToE fields (Figures 2b,e,h)
							  also save supplementary figure 9
	- process-0D_model_phyto_frac.py		: zero-dimensional water parcel model
							  also produces timeseries os isotope changes in schematic of Figure 3d
							  also save supplementary figure 8

Ferret scripts are named according to their action. 
They perform key processing actions for analysis or save new netcdf files for plotting in python.

Section 1: "Anthropogenic alteration"
	- ncycle_change.jnl			: calculates changes in N cycle variables shown in Figure 1
	- sources_and_sinks.jnl			: calculates changes in N cycle budget shown in Figure 1g

Section 2: "Isotopic signals of the anthropogenic alteration"
	- d15n.jnl				: calculates isotopic values of all nitrogen cycle variables
	- define_twilight.jnl			: defines the euphotic and twilight zone masks used to calculate average trends
	- isotopic_trends_depthzones.jnl	: calculates the mean isotopic trends in different depth zones (Figure 2a,d,g, 3a)
	- prep_files_for_ToE.jnl		: calculates mean trends of variables in depth zones for ToE analysis 
	- calculate_toe_percentcover.jnl	: following "process-compute_ToE.py", calculates percent cover of ocean of ToE
	- make_toe_curves.sh			: bash script to create cumulative ToE curves at each year in simulation (2c,f,i)

Section 3: "Linking climate change, d15N and nitrogen cycling"
	- decomposition_no3.jnl 		: isotopic flux analysis mentioned in first paragraph and saves results
	- fluxanalysis_results.jnl		: puts isotopic flux analysis results into depth brackets (EZ and TW) (Figure 3b)
	- figure2D_cc_din_e15n.jnl		: makes 2D fields needed for plotting (figure 3c)
	- assess_direct_indirect_effect.jnl	: computes anomalies in important N cycle variables from experiments that
						  isolated the direct and indirect effects of climate change (figure 4)

Conclusion:
	- find_data_count_twilightzone.jnl	: calculates the total number of observations in the twilight zone of the 
						  Pacific and Atlantic Oceans


For any questions, please email me at pearse.buchanan@liverpool.ac.uk | pearse.buchanan@hotmail.com

