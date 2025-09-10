import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.mask import mask
from rasterio.windows import from_bounds
import math

def reproj_match(infile, match, outfile, resampling_method='bilinear', save_in='asc'):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """

    if resampling_method == 'bilinear':
        resample_as = Resampling.bilinear
    if resampling_method == 'nearest':
        resample_as = Resampling.nearest
    if resampling_method == 'average':
        resample_as = Resampling.average
    if resampling_method == 'min':
        resample_as = Resampling.min

    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            dst_nodata = match.meta['nodata']
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": dst_nodata})
        if save_in=='asc':
            dst_kwargs.update({"driver": "AAIGrid"})
            
        print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resample_as)



def fill_layer_na_with_layer(priority_layer, secondary_layer, out_fp, save_in='geotiff'):

    with rasterio.open(priority_layer) as src1:
        data1 = src1.read(1)
        data1 = data1.astype(float)
        meta1 = src1.meta.copy()
        nodata1 = meta1['nodata']
        
    with rasterio.open(secondary_layer) as src2:
        data2 = src2.read(1)
        data2 = data2.astype(float)
        meta2 = src2.meta.copy()
        nodata2 = meta2['nodata']
        
    data1[data1 == nodata1] = data2[data1 == nodata1]
    data1 = data1.astype(int)

    out_meta = meta2.copy()
    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})  
        
    with rasterio.open(out_fp, 'w+', **out_meta) as out:
            src = out.write(data1, 1)                


def rasterize_shapefile(shapefile, burn_field, out_fp, ref_raster, subset=None, plot=True, save_in='geotiff'):

    fields = [burn_field, 'geometry']
    shape = gpd.read_file(shapefile, include_fields=fields, bbox=subset)

    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    meta.update({"nodata": -9999})

    shape[burn_field] = shape[burn_field].astype("float64")

    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
        
    with rasterio.open(out_fp, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(shape.geometry, shape[burn_field]))

        burned = features.rasterize(shapes=shapes, fill=-9999, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        #for key in mpk.keys():
        #    burned[burned == key] = mpk[key]
        out.write_band(1, burned)
        if plot==True:
            raster = rasterio.open(out_fp)
            show(raster)

def extract_and_clip_gpkg_layers(gpkg_file, ref_raster, out_dir):
    """
    Extracts and clips selected layers from a GeoPackage using the extent of a reference raster.
    
    Parameters:
        gpkg_file (str): Path to the GeoPackage file.
        ref_raster (str): Path to the reference raster (used for extent and CRS).
        out_dir (str): Directory where the clipped shapefiles will be saved.
    """
    # Layers to extract: {GPKG layer name: output shapefile name}
    layers = {
        'vaglinje': 'vaglinje',
        'ovrig_vag': 'ovrig_vag',
        'ralstrafik': 'ralstrafik',
    }

    # Read raster extent and CRS
    with rasterio.open(ref_raster) as src:
        bounds = src.bounds
        raster_crs = src.crs
        raster_box = box(*bounds)  # shapely box
        raster_gdf = gpd.GeoDataFrame(geometry=[raster_box], crs=raster_crs)

    # Process each layer
    for layer_name, out_name in layers.items():
        print(f"Processing layer: {layer_name}")
        
        # Read the layer
        gdf = gpd.read_file(gpkg_file, layer=layer_name)

        # Reproject to raster CRS if needed
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Clip to raster extent
        clipped = gpd.clip(gdf, raster_gdf)

        # Save to shapefile
        out_path = os.path.join(out_dir, f"{out_name}.shp")
        clipped.to_file(out_path)

        print(f"Saved clipped shapefile to: {out_path}")


def rasterize_water_bodies(stream_file=None, river_file=None, lake_file=None, ref_raster=None):
    """
    Rasterize stream, river, and/or lake geometries into the grid of a reference DEM.
    Any of the vector layers (stream, river, lake) can be omitted by passing None.

    Parameters:
        stream_file (str or None): Path to stream shapefile (lines).
        river_file (str or None): Path to river shapefile (polygons).
        lake_file (str or None): Path to lake shapefile (polygons).
        ref_raster (str): Path to reference raster (e.g., DEM).

    Returns:
        str: Path to the output hydrological mask raster.
    """
    if ref_raster is None:
        raise ValueError("ref_raster must be provided.")

    base, _ = os.path.splitext(ref_raster)

    suffix_parts = []
    if stream_file: suffix_parts.append("streams")
    if river_file:  suffix_parts.append("rivers")
    if lake_file:   suffix_parts.append("lakes")
    suffix = "_".join(suffix_parts) or "water"

    out_raster = f"{base}_{suffix}.tif"

    # Load and collect vector data
    vector_layers = []
    for file in [stream_file, river_file, lake_file]:
        if file:
            gdf = gpd.read_file(file)
            vector_layers.append(gdf)

    if not vector_layers:
        raise ValueError("At least one of stream_file, river_file, or lake_file must be provided.")

    # Open reference raster
    with rasterio.open(ref_raster) as ref:
        meta = ref.meta.copy()
        out_shape = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    # Reproject all to raster CRS
    for gdf in vector_layers:
        if gdf.crs != crs:
            gdf.to_crs(crs, inplace=True)

    # Collect all geometries
    combined_shapes = []
    for gdf in vector_layers:
        combined_shapes.extend([(geom, 1) for geom in gdf.geometry if geom is not None])

    # Rasterize
    rasterized = rasterize(
        combined_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Update metadata
    meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "compress": "lzw",
        "nodata": 0
    })

    with rasterio.open(out_raster, 'w', **meta) as dst:
        dst.write(rasterized, 1)

    print(f"Hydrological mask raster saved to: {out_raster}")
    return out_raster


def burn_water_into_dem(dem_path, water_mask_path, drop=1.0):
    """
    Lowers DEM elevation by `drop` meters where water bodies are present in the water mask.

    Parameters:
        dem_path (str): Path to the input DEM (.tif).
        water_mask_path (str): Path to the binary water mask raster (non-zero values indicate water).
        drop (float): Amount in meters to drop elevation in water-covered cells.

    Returns:
        str: Path to the modified DEM with "_water" suffix.
    """
    # Load DEM
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_nodata = dem_src.nodata

    # Load water mask
    with rasterio.open(water_mask_path) as water_src:
        water_data = water_src.read(1)

    # Identify water cells
    water_mask = water_data > 0

    # Modify DEM
    dem_modified = np.copy(dem_data)
    if dem_nodata is not None:
        valid_mask = (dem_data != dem_nodata)
        dem_modified[water_mask & valid_mask] -= drop
    else:
        dem_modified[water_mask] -= drop

    # Prepare metadata
    dem_meta.update({
        'dtype': 'float32',
        'nodata': dem_nodata,
        'compress': 'lzw'
    })

    # Construct output path
    base, ext = os.path.splitext(dem_path)
    output_path = f"{base}_water{ext}"

    # Save modified DEM
    with rasterio.open(output_path, 'w', **dem_meta) as dst:
        dst.write(dem_modified.astype('float32'), 1)

    print(f"Modified DEM saved to: {output_path}")
    return output_path

# --- Helper function ---
def coords_to_index(transform, x, y):
    return rowcol(transform, x, y)

def clip_raster_to_catchment(input_fp, catchment_fp, output_fp, nodata_val):
    """Clip raster to catchment shapefile"""
    with rasterio.open(input_fp) as src:
        gdf = gpd.read_file(catchment_fp).to_crs(src.crs)
        out_image, out_transform = mask(src, gdf.geometry, crop=True, nodata=nodata_val)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_val
        })

    with rasterio.open(output_fp, 'w', **out_meta) as dest:
        dest.write(out_image)


def clip_raster_to_catchment(input_fp, catchment_fp, output_fp, nodata_val):
    with rasterio.open(input_fp) as src:
        gdf = gpd.read_file(catchment_fp).to_crs(src.crs)
        out_image, out_transform = mask(src, gdf.geometry, crop=True, nodata=nodata_val)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_val
        })

    with rasterio.open(output_fp, 'w', **out_meta) as dest:
        dest.write(out_image)

def clip_vector_to_catchment(input_fp, catchment_fp, output_fp):
    gdf = gpd.read_file(input_fp)
    catchment = gpd.read_file(catchment_fp)

    if gdf.crs != catchment.crs:
        gdf = gdf.to_crs(catchment.crs)

    clipped = gpd.overlay(gdf, catchment, how='intersection')
    clipped.to_file(output_fp)


def open_raster_with_subset(fp, out_fp, subset, plot=True, save_in='asc'):
    
    with rasterio.open(fp) as dataset:
        data = dataset.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], dataset.transform))
        data = data.astype(float)
        out_meta = dataset.profile.copy()
        
    new_affine = rasterio.Affine(out_meta['transform'][0], 
                                out_meta['transform'][1], 
                                subset[0], 
                                out_meta['transform'][3], 
                                out_meta['transform'][4], 
                                subset[3])   

    out_meta.update({"height": data.shape[0],
                      "width": data.shape[1],
                      "transform": new_affine,
                    }
                    )

    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})  
        
    with rasterio.open(out_fp, 'w', **out_meta) as dst:
        src = dst.write(data, 1)
            
    raster = rasterio.open(out_fp)
    if plot==True:
        show(raster)

def fill_layer_na_with_layer(priority_layer, secondary_layer, out_fp, save_in='geotiff'):

    with rasterio.open(priority_layer) as src1:
        data1 = src1.read(1)
        data1 = data1.astype(float)
        meta1 = src1.meta.copy()
        nodata1 = meta1['nodata']
        
    with rasterio.open(secondary_layer) as src2:
        data2 = src2.read(1)
        data2 = data2.astype(float)
        meta2 = src2.meta.copy()
        nodata2 = meta2['nodata']
        
    data1[data1 == nodata1] = data2[data1 == nodata1]
    data1 = data1.astype(int)

    out_meta = meta2.copy()
    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})  
        
    with rasterio.open(out_fp, 'w+', **out_meta) as out:
            src = out.write(data1, 1)

def ba_from_vol(vol, a):
    ba = a * vol
    return ba

def cf_from_ba(ba):
    cf = 0.1939 * ba / (0.1939 * ba + 1.69)
    return cf


def stem_volume_to_LAI(V, tree='spruce'):
    '''
    Based on Lehtonen et al. 2007: Biomass expansion factors (BEFs) for Scots pine, 
    Norway spruce and birch according to stand age for boreal forests
    
    V = volume [m3 ha-1]
    tree = 'spruce' / 'pine' / 'birch'
    '''
    
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    if tree == 'pine':
        a = math.exp(-2.2532) # ln(a)
        b = 0.7802
    if tree == 'spruce':
        a = math.exp(-1.4772)
        b = 0.7718
    if tree == 'decid':
        a = 0.02542
        b = 0.96

    W_V = a * V**b # Mg ha-1
    W_V = W_V * 1e3 # kg ha-1
    W_V = W_V * 1e-4 # kg m-2
    
    LAI = W_V * SLA[tree] # m3 m-3
    
    return LAI


def stem_volume_to_LAI_mVMI_nonlinear(V, tree='spruce'):
    '''
    Based on S.L. fitted nonlinear models using mVMI data on mineral soil
    for pine, spruce and deciduous
    
    V = volume [m3 ha-1]
    tree = 'spruce' / 'pine' / 'birch'
    '''
    
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    if tree == 'pine':
        a = -2.94e1
        b = 4.70e0
        c = -7.63e-3
    if tree == 'spruce':
        a = -7.01e1
        b = 2.16e1
        c = -3.43e-3
    if tree == 'decid':
        a = -7.55e1
        b = 4.25e0
        c = -1.42e-2

    W_V = a + b * (1 - math.exp(c * V)) # 1000 kg ha-1
    W_V = W_V * 1e3 # kg ha-1
    W_V = W_V * 1e-4 # kg m-2
    
    LAI = W_V * SLA[tree] # m3 m-3
    
    return LAI

def stem_volume_to_LAI_mVMI_nonlinear_2(V, tree='spruce'):
    '''
    Based on S.L. fitted nonlinear models using mVMI data on mineral soil
    for pine, spruce and deciduous
    
    V = volume [m3 ha-1]
    tree = 'spruce' / 'pine' / 'birch'
    '''
    
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    if tree == 'pine':
        a = -4.79e0
        b = -6.08e-3
    if tree == 'spruce':
        a = 2.22e1
        b = -2.97e-3
    if tree == 'decid':
        a = 3.93e0
        b = -9.28e-3

    W_V = a * (1 - math.exp(b * V)) # 1000 kg ha-1
    W_V = W_V * 1e3 # kg ha-1
    W_V = W_V * 1e-4 # kg m-2
    
    LAI = W_V * SLA[tree] # m3 m-3
    
    return LAI

def stem_volume_to_LAI_mVMI_fit(V, tree='spruce'):
    '''
    Based on Lehtonen et al. 2007: Biomass expansion factors (BEFs) for Scots pine, 
    Norway spruce and birch according to stand age for boreal forests
    Parameters optimized by Samuli with Hyytiälä mVMI data.
    
    V = volume [m3 ha-1]
    tree = 'spruce' / 'pine' / 'birch'
    '''
    
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    if tree == 'pine':
        a = 0.03768
        b = 0.86
    if tree == 'spruce':
        a = 0.12304
        b = 0.81
    if tree == 'decid':
        a = 0.02542
        b = 0.96

    W_V = a * V**b # Mg ha-1
    W_V = W_V * 1e3 # kg ha-1
    W_V = W_V * 1e-4 # kg m-2
    
    LAI = W_V * SLA[tree] # m3 m-3
    
    return LAI