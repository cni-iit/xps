using Printf
using Plots
using DataFrames
using CSV

"""
    Spectrum

A data structure to hold XPS spectrum data including metadata and measurement points.

# Fields
- `metadata::Dict{String, String}`: Dictionary of metadata about the spectrum
- `binding_energy::Vector{Float64}`: Vector of binding energy values in eV
- `counts_per_second::Vector{Float64}`: Vector of corresponding intensity values
- `spectrum_id::Union{String, Nothing}`: Optional identifier for the spectrum
- `region::Union{String, Nothing}`: Optional region name for the spectrum
"""
mutable struct Spectrum
    metadata::Dict{String, String}
    binding_energy::Vector{Float64}
    counts_per_second::Vector{Float64}
    spectrum_id::Union{String, Nothing}
    region::Union{String, Nothing}
    
    # Constructor with default values
    function Spectrum()
        return new(Dict{String, String}(), Float64[], Float64[], nothing, nothing)
    end
end

"""
    plot(spectrum::Spectrum; kwargs...)

Plot a spectrum with appropriate labels and XPS conventions.

# Arguments
- `ax=nothing`: Optional existing plot to add this spectrum to
- `label=nothing`: Optional label for the plot legend
- Additional keyword arguments are passed to the plotting function

# Returns
- The plot object
"""
function plot(spectrum::Spectrum; ax=nothing, label=nothing, kwargs...)
    # Create a new plot if none is provided
    if ax === nothing
        p = plot(size=(800, 500))
    else
        p = ax
    end
    
    # Create default label if none provided
    if label === nothing
        if spectrum.region !== nothing && spectrum.spectrum_id !== nothing
            label = "$(spectrum.region) (ID: $(spectrum.spectrum_id))"
        elseif spectrum.region !== nothing
            label = spectrum.region
        elseif spectrum.spectrum_id !== nothing
            label = "Spectrum ID: $(spectrum.spectrum_id)"
        else
            label = "Spectrum"
        end
    end
    
    # Plot the data
    plot!(p, spectrum.binding_energy, spectrum.counts_per_second, 
        label=label, 
        xlabel="Binding Energy (eV)",
        ylabel="Counts per Second",
        xflip=true,  # XPS convention: binding energy decreases from left to right
        kwargs...
        )
    
    return p
end

"""
    XPSData

A data structure to hold multiple XPS spectra with associated metadata.

# Fields
- `general_metadata::Dict{String, String}`: General metadata about the dataset
- `spectra::Vector{Spectrum}`: Vector of spectra in the dataset
"""
mutable struct XPSData
    general_metadata::Dict{String, String}
    spectra::Vector{Spectrum}
    
    # Constructor with default values
    function XPSData()
        return new(Dict{String, String}(), Spectrum[])
    end
end

"""
    parse_file(xps_data::XPSData, file_path::String)

Parse an XPS .xy file and load the data into an XPSData structure.

# Arguments
- `file_path`: Path to the .xy file to parse

# Returns
- The updated XPSData object
"""
function parse_file(xps_data::XPSData, file_path::String)
    lines = readlines(file_path)
    
    # Process the file
    current_section = "general_metadata"
    current_spectrum = nothing
    
    for line in lines
        line = strip(line)
        
        # Skip empty lines
        if isempty(line)
            continue
        end
        
        # Check if we're in the metadata section
        if startswith(line, '#')
            # Remove the '# ' prefix if it exists
            cleaned_line = startswith(line, "# ") ? line[3:end] : line[2:end]
            
            # Check for section transitions
            if occursin("Region:", cleaned_line)
                # We're starting a new spectrum section
                if current_spectrum !== nothing
                    push!(xps_data.spectra, current_spectrum)
                end
                
                current_spectrum = Spectrum()
                current_section = "spectrum_metadata"
            end
            
            # Process metadata based on current section
            if current_section == "general_metadata"
                _process_metadata_line(cleaned_line, xps_data.general_metadata)
            elseif current_section == "spectrum_metadata" && current_spectrum !== nothing
                _process_metadata_line(cleaned_line, current_spectrum.metadata)
                
                # Extract important values
                if occursin("Region:", cleaned_line)
                    parts = split(cleaned_line, ":", limit=2)
                    if length(parts) > 1
                        current_spectrum.region = strip(parts[2])
                    end
                elseif occursin("Spectrum ID:", cleaned_line)
                    parts = split(cleaned_line, ":", limit=2)
                    if length(parts) > 1
                        current_spectrum.spectrum_id = strip(parts[2])
                    end
                end
            end
            
            # Check if we're transitioning to data section
            if occursin("ColumnLabels:", cleaned_line)
                current_section = "data"
            end
        
        # Process data points
        elseif current_section == "data" && current_spectrum !== nothing
            try
                parts = split(line)
                if length(parts) >= 2
                    binding_energy = parse(Float64, parts[1])
                    counts = parse(Float64, parts[2])
                    push!(current_spectrum.binding_energy, binding_energy)
                    push!(current_spectrum.counts_per_second, counts)
                end
            catch
                # Skip lines that can't be parsed as data points
            end
        end
    end
    
    # Add the last spectrum if it exists
    if current_spectrum !== nothing
        push!(xps_data.spectra, current_spectrum)
    end
    
    return xps_data
end

"""
    _process_metadata_line(line::String, metadata_dict::Dict{String, String})

Process a metadata line and add it to the appropriate dictionary.

# Arguments
- `line`: The metadata line to process
- `metadata_dict`: The dictionary to add the metadata to
"""
function _process_metadata_line(line::String, metadata_dict::Dict{String, String})
    if occursin(":", line)
        parts = split(line, ":", limit=2)
        metadata_dict[strip(parts[1])] = strip(parts[2])
    end
end

"""
    plot_all_spectra(xps_data::XPSData; title=nothing, figsize=(900, 600))

Plot all spectra in the dataset.

# Arguments
- `title`: Optional title for the plot
- `figsize`: Optional size for the figure

# Returns
- The plot object
"""
function plot_all_spectra(xps_data::XPSData; title=nothing, figsize=(900, 600))
    p = plot(size=figsize)
    
    for spectrum in xps_data.spectra
        plot(spectrum, ax=p)
    end
    
    if title !== nothing
        plot!(p, title=title)
    else
        plot!(p, title="XPS Spectra")
    end
    
    return p
end

"""
    get_spectrum_by_id(xps_data::XPSData, spectrum_id)

Get a spectrum by its ID.

# Arguments
- `spectrum_id`: The ID to search for

# Returns
- The spectrum if found, nothing otherwise
"""
function get_spectrum_by_id(xps_data::XPSData, spectrum_id)
    for spectrum in xps_data.spectra
        if spectrum.spectrum_id == string(spectrum_id)
            return spectrum
        end
    end
    return nothing
end

"""
    get_spectrum_by_region(xps_data::XPSData, region_name::String)

Get spectra by region name.

# Arguments
- `region_name`: The region name to search for

# Returns
- A vector of matching spectra
"""
function get_spectrum_by_region(xps_data::XPSData, region_name::String)
    return filter(s -> s.region == region_name, xps_data.spectra)
end

"""
    save_to_csv(xps_data::XPSData, output_dir::String, base_filename=nothing)

Save all spectra to individual CSV files.

# Arguments
- `output_dir`: Directory to save the CSV files
- `base_filename`: Optional base name for the CSV files
"""
function save_to_csv(xps_data::XPSData, output_dir::String, base_filename=nothing)
    # Create the output directory if it doesn't exist
    mkpath(output_dir)
    
    if base_filename === nothing
        base_filename = "spectrum"
    end
    
    for (i, spectrum) in enumerate(xps_data.spectra)
        # Create a meaningful filename
        if spectrum.region !== nothing && spectrum.spectrum_id !== nothing
            filename = "$(base_filename)_$(spectrum.region)_$(spectrum.spectrum_id).csv"
        elseif spectrum.region !== nothing
            filename = "$(base_filename)_$(spectrum.region)_$(i).csv"
        elseif spectrum.spectrum_id !== nothing
            filename = "$(base_filename)_$(spectrum.spectrum_id).csv"
        else
            filename = "$(base_filename)_$(i).csv"
        end
        
        # Replace any characters that might not be valid in filenames
        filename = replace(filename, r"[^\w\.-]" => "_")
        
        # Write to CSV
        filepath = joinpath(output_dir, filename)
        
        # Create a DataFrame for the data
        df = DataFrame(
            "Binding Energy" => spectrum.binding_energy,
            "Counts per Second" => spectrum.counts_per_second
        )
        
        # Write metadata as comments in a separate operation
        open(filepath, "w") do f
            # Write metadata as comments
            write(f, "# XPS Spectrum Data\n")
            for (key, value) in spectrum.metadata
                write(f, "# $(key): $(value)\n")
            end
        end
        
        # Append the data frame
        CSV.write(filepath, df, append=true)
    end
end

"""
    compare_spectra(xps_data::XPSData, region_names::Vector{String}; normalize=false)

Plot multiple spectra from different regions for comparison.

# Arguments
- `region_names`: Vector of region names to include
- `normalize`: Whether to normalize the spectra

# Returns
- The plot object
"""
function compare_spectra(xps_data::XPSData, region_names::Vector{String}; normalize=false)
    p = plot(size=(900, 600))
    
    for region in region_names
        spectra = get_spectrum_by_region(xps_data, region)
        for spectrum in spectra
            # Make a copy of the data for plotting
            x = spectrum.binding_energy
            y = copy(spectrum.counts_per_second)
            
            # Normalize if requested
            if normalize && maximum(y) > 0
                y = y ./ maximum(y)
            end
            
            label = "$(spectrum.region) (ID: $(spectrum.spectrum_id))"
            plot!(p, x, y, label=label)
        end
    end
    
    plot!(p, 
        xlabel="Binding Energy (eV)",
        ylabel=normalize ? "Normalized Intensity" : "Counts per Second",
        title="Comparison of Different XPS Regions",
        xflip=true)  # XPS convention
    
    return p
end

"""
    calculate_peak_area(spectrum::Spectrum, start_energy::Float64, end_energy::Float64)

Calculate the area under a peak within a specific energy range.

# Arguments
- `start_energy`: The starting energy of the range
- `end_energy`: The ending energy of the range

# Returns
- The calculated area
"""
function calculate_peak_area(spectrum::Spectrum, start_energy::Float64, end_energy::Float64)
    # Find indices corresponding to the energy range
    indices = findall(e -> start_energy <= e <= end_energy, spectrum.binding_energy)
    
    if isempty(indices)
        return 0.0
    end
    
    # Extract the relevant energy and intensity values
    energies = spectrum.binding_energy[indices]
    intensities = spectrum.counts_per_second[indices]
    
    # Sort by energy if needed
    if energies[1] < energies[end]
        sorted_indices = sortperm(energies)
        energies = energies[sorted_indices]
        intensities = intensities[sorted_indices]
    end
    
    # Calculate area using trapezoidal rule
    area = 0.0
    for i in 1:(length(energies)-1)
        area += (intensities[i] + intensities[i+1]) * abs(energies[i] - energies[i+1]) / 2
    end
    
    return area
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Replace with your actual file path
    file_path = "prova.xy"
    
    xps_data = XPSData()
    parse_file(xps_data, file_path)
    
    println("Loaded $(length(xps_data.spectra)) spectra")
    println("\nGeneral Metadata:")
    for (key, value) in xps_data.general_metadata
        println("  $(key): $(value)")
    end
    
    # Example: Plot all spectra
    p1 = plot_all_spectra(xps_data)
    display(p1)
    savefig(p1, "all_spectra.png")
    
    # Example: Get a specific spectrum and plot it
    spec = get_spectrum_by_region(xps_data, "Survey")
    if !isempty(spec)
        p2 = plot(spec[1])
        plot!(p2, title="Survey Spectrum (ID: $(spec[1].spectrum_id))")
        display(p2)
        savefig(p2, "survey_spectrum.png")
    end
    
    # Example: Save to CSV
    save_to_csv(xps_data, "output_data")
    
    # Example: Compare multiple spectra
    # Uncomment to use:
    # p3 = compare_spectra(xps_data, ["C1s", "O1s", "N1s"], normalize=true)
    # display(p3)
    # savefig(p3, "spectrum_comparison.png")
    
    # Example: Calculate peak area
    # Uncomment to use:
    # spec = get_spectrum_by_region(xps_data, "C1s")
    # if !isempty(spec)
    #     area = calculate_peak_area(spec[1], 285.0, 288.0)
    #     println("Area under C1s peak between 285-288 eV: $(round(area, digits=2))")
    # end
end