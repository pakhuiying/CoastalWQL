
import PySimpleGUI as sg
from preprocessing import *
from base64_icons import *
# venv: GUI_test_python36

def draw_plot(image_file_path):
    '''
    Plot GPS coordinates using matplotlib GUI interface
    '''
    gps_df = import_gps(image_file_path)#
    unique_df = get_unique_df(gps_df)
    fig, ax = plt.subplots()
    ax.set_title('click on points to select start and end points')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    #add 
    global coords
    coords = []
    global indices
    indices = []

    line, = ax.plot(unique_df['longitude'],unique_df['latitude'],'o', #will plot a scatter plot when 'o' is used
                    picker=True, pickradius=5)  # 5 points tolerance


    def onpick(event):
        thisline = event.artist

        # global xdata, ydata
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()

        # global ind
        ind = event.ind

        coords.append((xdata[ind[0]], ydata[ind[0]]))

        indices.append(ind[0])

        print("Coords: {}\nIndices: {}".format(coords,indices))
        ax.plot(xdata[ind[0]], ydata[ind[0]],'ro')
        fig.canvas.draw()

        if values['-PROCESSED_IMAGES-'] == '':
            gps_fp = join(values['-IMAGE_FOLDER_FILEPATH-'],"gps_index_{}.txt".format(values['-PREFIX-']))
        else:
            gps_fp = join(values['-PROCESSED_IMAGES-'],"gps_index_{}.txt".format(values['-PREFIX-']))
        # gps_fp = join(image_file_path,"gps_index_{}.txt".format(values['-PREFIX-']))
        with open(gps_fp, "w") as output:
            for i in indices:
                output.write(str(i)+'\n') #write list of gps indices to .txt file
        return coords, indices


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block=False)
    return indices

sg.ChangeLookAndFeel('Default1')    
# -------------------------- plt interface for plotting gps coord ---------------------------


# --------------------------------- Define Layout ---------------------------------
def get_main_window():

    # # ----- Full layout -----
    wavelengths = bands_wavelengths() # list of wavelength
    wavelengths = ['{:.2f}'.format(i) for i in wavelengths]
    bands = [f' (Band{i})' for i in range(len(wavelengths))]
    dropdown_list = tuple([w+b for w, b in zip(wavelengths, bands)])

    tab1_layout =  [[sg.Text('Image folder path:'), sg.Text(size=(25,1))],
                    [sg.Input(size=(50,1),  key='-IMAGE_FOLDER_FILEPATH-'), sg.FolderBrowse()],
                    [sg.Text('Spectrometer folder path:'), sg.Text(size=(25,1))],
                    [sg.Input(size=(50,1), key='-SPECTRO_FILEPATH-'), sg.FolderBrowse()],
                    [sg.Text('Prefix for files:')],
                    [sg.Input(size=(50,1),default_text='input_prefix',key='-PREFIX-',tooltip='Set a different prefix when processing a different set of images e.g. datetime')],
                    [sg.Text('Location to store processed images:'), sg.Text(size=(25,1))],
                    [sg.Input(size=(50,1),key='-PROCESSED_IMAGES-'), sg.FolderBrowse()],
                    [sg.Text('Or upload previous configuration file:'), sg.Button('Upload config file',key="-CONFIG_FILEPATH-")]
                    ]

    tab2_layout = [[sg.Frame(layout=[      
                    [sg.Push(),sg.Button('Launch to select region',tooltip='Click on Launch button to open up another GUI window'),sg.Push()],
                    [sg.Text('Drone height (m):',size=(12,1)),sg.Input(size=(5,1),key='-HEIGHT-')],
                    [sg.Text('Upload .txt file:',size=(12,1)),sg.Input(size=(33,1),tooltip='gps txt file is stored in the processed location defined above or under image folder',enable_events=True,key='-GPS_INDEX_TXT-'), sg.FileBrowse()],
                    [sg.Text('GPS indices:',size=(10,1)), sg.Text(size=(25,1),enable_events=True,key='-GPS_INDEX-')]
                    ], title='Specify flight region',font='Arial 8 bold',size=(460,150))],

                    [sg.Frame(layout=[      
                    [sg.Text(size=(30,1),enable_events=True,key='-NUMBER_OF_LINES-')],
                    [sg.Push(),sg.Text('Line start:'),sg.Spin(values=[], initial_value=0, k='-LINE_START-'),sg.Text('Line end:'),sg.Spin(values=[], k='-LINE_END-'),sg.Push()],
                    ], title='Range of lines to process',font='Arial 8 bold',size=(460,80))],

                    [sg.Frame(layout=[      
                    [sg.Push(),sg.Button('Launch image correction',key="-CORRECTION-"),sg.Push()],
                    [sg.Text('Correct images: '),sg.Button('Upload corrected indices',key='-CORRECTED_IMG_INDICES-',tooltip="Saved .json file from launching image correction")],
                    [sg.Push(),sg.Slider(range=(0,2000),default_value=1400,size=(40,15),orientation='h', key='-SLIDER-'),sg.Push()]
                    ], title='Image alignment correction (msec)',font='Arial 8 bold',size=(460,130))],

                    ]    

    tab3_layout = [[sg.Text('Water quality .csv file',size=(17,1)),sg.Input(size=(30,1),enable_events=True,key='-WQL_CSV-'), sg.FileBrowse()],
                    [sg.Text('Column of latitude',size=(17,1)),sg.DropDown(values=[],size=(20,1),key='-LAT_COL-')],
                    [sg.Text('Column of longitude',size=(17,1)),sg.DropDown(values=[],size=(20,1),key='-LON_COL-')],
                    [sg.Text('Column of wql variable',size=(17,1)),sg.DropDown(values=[],size=(20,1),key='-WQL_COL-')],
                    

                    [sg.Frame(layout=[
                    [sg.Multiline('', size=(35,5), expand_x=True, expand_y=True, k='-QUERY_DF-',tooltip='Refer to documentation of querying language'),
                    sg.DropDown(values=[], enable_events=True,key='-QUERY_VARIABLES-',size=(20,1),tooltip='Select variable to add in the query box')],
                    [sg.Button('View water quality file')],
                    ], title='Querying & Filtering',font='Arial 8 bold',size=(460,140))]
                    ]

    tab4_layout = [[sg.Frame(layout=[
                    [sg.Checkbox('Mask objects',size=(12,1),enable_events=True,key='-MASK_CHECKBOX-'),
                    sg.Checkbox('Classify objects',size=(17,1),key='-CLASSIFY_CHECKBOX-')],
                    [sg.Checkbox('Noise removal',default=True,size=(12,1),key='-NOISE_CHECKBOX-',tooltip="Destriping to remove stripe noises"),
                    sg.Checkbox('Radiometric correction',default=True,size=(17,1),key='-RADIOMETRIC_CHECKBOX-')],
                    [sg.Checkbox('Sunglint correction',default=True,size=(17,1),key='-SUNGLINT_CHECKBOX-')],
                    [sg.Text('Select pseudo-RGB bands:')],
                    [sg.Text('R'),sg.DropDown(values=dropdown_list, enable_events=True,key='-R-', size=(13,1)),
                    sg.Text('G'),sg.DropDown(values=dropdown_list, enable_events=True,key='-G-', size=(13,1)),
                    sg.Text('B'),sg.DropDown(values=dropdown_list, enable_events=True,key='-B-', size=(13,1))]
                    ], title='Preprocessing',font='Arial 8 bold',size=(460,160))],

                    [sg.Frame(layout=[
                    [sg.Checkbox('Generate prediction map',size=(20,1),enable_events=True,key='-PREDICT_CHECKBOX-')],
                    [sg.Input(size=(50,1), default_text='File path of model', enable_events=True,key='-MODEL_FILEPATH-'), sg.FileBrowse()],
                    [sg.Text('List of bands as predictors:')],
                    [sg.Input(size=(50,1), key='-MODEL_PREDICTORS-',tooltip='INT input only! Use \':\' to indicate range and \',\' to indicate individual variable')],
                    [sg.Text('Downsampling factor:'),sg.Slider(range=(0,100),default_value=40,size=(15,15),orientation='h', key='-DOWNSAMPLE_SLIDER-')]
                    ], title='Prediction',font='Arial 8 bold',size=(460,210))]]

    tab5_layout = [
                    [sg.Text('Preview previously processed images:')],
                    [sg.Input(size=(50,1),default_text='\'Processed\' Image Folder to open',enable_events=True,key='-BROWSE_IMAGES-'), sg.FolderBrowse()],
                    [sg.DropDown(values=[], enable_events=True, key='-PROCESSED_GEOTRANSFORMED_IMAGES-',size=(40,1)),
                    sg.Button('View georeferenced images')],
                    [sg.Text('Assess extracted spectral information:')],
                    [sg.DropDown(values=[], key='-PROCESSED_WQL_CSV-',size=(40,1),tooltip='Select variable to add in the query box'),
                    sg.Button('Plot spectral curve',tooltip='Button will be enabled after browsing of processed images')],
                    [sg.Text('Tool to create various pseudo-RGB images for better visualisation:')],
                    [sg.Push(),sg.Button('Select hyperspectral bands'),sg.Push()],
                    ]

    tab6_layout = [
                    [sg.Text('List of dates:',size=(12,1))],
                    [sg.Multiline('', size=(35,2), expand_x=True, expand_y=True, k='-DATE_LIST-',tooltip='List of dates,separated by ;'),
                    sg.B('Select Date',key="-SELECT_DATE-")],
                    [sg.Text('List of start time:',size=(12,1))],
                    [sg.Multiline('', size=(35,2), expand_x=True, expand_y=True, k='-START_TIME_LIST-',tooltip='List of time in HH-MM-SS,separated by ;')],
                    [sg.Text('List of end time:',size=(12,1))],
                    [sg.Multiline('', size=(35,2), expand_x=True, expand_y=True, k='-END_TIME_LIST-',tooltip='List of time in HH-MM-SS,separated by ;')],
                    [sg.Text('Environmental parameters:',size=(20,1))],
                    [sg.Checkbox('Wind direction',default=False,size=(15,1),key='-WIND_DIRECTION_CHECKBOX-'),
                    sg.Checkbox('Wind speed',default=False,size=(15,1),key='-WIND_SPEED_CHECKBOX-')],
                    [sg.Checkbox('Air temperature',default=False,size=(15,1),key='-AIR_TEMPERATURE_CHECKBOX-'),
                    sg.Checkbox('Relative humidity',default=False,size=(15,1),key='-RELATIVE_HUMIDITY_CHECKBOX-')],
                    [sg.Text('Select location:',size=(20,1))],
                    [sg.DropDown(values=list(get_env_locations().values()), key='-ENV_LOCATION-',size=(40,1),tooltip='Select location')],
                    [sg.B('Fetch data',key="-FETCH_ENV_DATE-"),sg.B('Plot environmental data',key='-PLOT_ENV-')]
                    ]

    left_layout = [
            [sg.TabGroup([[sg.Tab('Required Inputs', tab1_layout), 
                sg.Tab('Region selection', tab2_layout),
                sg.Tab('WQL inputs', tab3_layout),
                sg.Tab('Options', tab4_layout),
                sg.Tab('Processed options', tab5_layout),
                sg.Tab('Env parameters', tab6_layout),
                ]],font='Arial 9 bold',pad=(5,10))],    
            [sg.Button('Process'), sg.Exit()]
            ]    

    images_list_col = [
                    [sg.Text(size=(45,1), key='-TOUT-')],
                    [sg.Text('Select processed image from the list:')],
                    [sg.Listbox(values=[], enable_events=True, horizontal_scroll=True,size=(40,15),key='-IMAGE LIST-')],
                    [sg.Button('Generate hyperspectral images')],
                    ]

    images_col = [               
                    [sg.Image(key='-IMAGE-')]
                    ]

    layout = [[sg.Column(left_layout, element_justification='l'), sg.Column(images_list_col, element_justification='c'), sg.Column(images_col, element_justification='c')]]
    # --------------------------------- Create Window ---------------------------------
    # window = sg.Window('stitching program',layout,resizable=True,finalize=True)
    return sg.Window('CoastalWQL',layout,resizable=True,finalize=True)

def get_table_window(data,header_list):
   
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  vertical_scroll_only=False,
                  display_row_numbers=True,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    return sg.Window('WQL Table', layout, grab_anywhere=False,resizable=True,finalize=True)

def get_hyperspectral_window():
    """
    Allows user to choose the three bands via 3 drop down list, and automatically updates the band
    """
    wavelengths = bands_wavelengths() # list of wavelength
    wavelengths = ['{:.2f}'.format(i) for i in wavelengths]
    bands = [f' (Band {i})' for i in range(len(wavelengths))]
    dropdown_list = tuple([w+b for w, b in zip(wavelengths, bands)])

    left_layout = [[sg.Input(size=(50,1),default_text='Hyperspectral folder to open',enable_events=True,key='-BROWSE_HYPERSPECTRAL_IMAGES-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(40,10),key='-HYPERSPECTRAL_IMAGE_LIST-')],
                ]
                

    right_layout = [[sg.Image(key='-HYPERSPECTRAL_IMAGE-')],
                [sg.Text('R'),sg.DropDown(values=dropdown_list, enable_events=True,key='-RED_BAND-', auto_size_text=True),
                sg.Text('G'),sg.DropDown(values=dropdown_list, enable_events=True,key='-GREEN_BAND-', auto_size_text=True),
                sg.Text('B'),sg.DropDown(values=dropdown_list, enable_events=True,key='-BLUE_BAND-', auto_size_text=True)]
                ]

    layout = [[sg.Column(left_layout, element_justification='c'), sg.Column(right_layout, element_justification='c')]]

    return sg.Window('Hyperspectral band selection', layout, grab_anywhere=False,resizable=True,finalize=True,element_justification='center')

def get_correction_canvas_window(g_size,canvas_scale,n_lines):
    """
    plot stitch images on tkinter canvas which allows for live correction
    g_size (tuple of integers): size of the graph in pixels
    n_lines (int): number of sliders to add to individually correct a line image
    """
    g_size_bigger = (int(canvas_scale*g_size[0]),int(canvas_scale*g_size[1]))

    sliders_col = [[sg.Col([[sg.Push(),sg.Text("Line {}".format(i)),sg.Push()],
    [sg.Slider(range=(0,2000),default_value=1400,size=(15,15),orientation='v',enable_events=True,key="-S_{}-".format(i))]
    ],element_justification="center") for i in range(n_lines)],
    [sg.Push(),sg.Button('Save correction',key='-SAVE_CORRECTION-'),sg.Push()]
    ]

    layout = [
        [sg.Button('', image_data=sg.red_x,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='Exit',tooltip='Exit',pad=((0,20),(0,0)))],   
        [sg.Graph(
            canvas_size=g_size,
            graph_bottom_left=(0, 0),
            graph_top_right=g_size_bigger,
            key="-CORRECTION_GRAPH-",
            enable_events=True,
            background_color='lightblue',
            drag_submits=True,
            
            ),
        sg.Col(sliders_col, key='-COL1-'),
            ],
        [sg.Slider(range=(0,2000),default_value=1400,size=(40,15),orientation='h',enable_events=True,key="-SLIDER_CORRECTION-")
        ],   
        [sg.Text(key='-CORRECTION_INFO-', size=(40, 1))]
        ]
    return sg.Window("Perform time correction", layout, no_titlebar=True,grab_anywhere=True,finalize=True)

def get_sgc_canvas():
    layout = [[sg.B('Exit'), sg.B('Proceed',key="-PROCEED_PROCESS-")],
            [sg.Canvas(key='controls_cv')],
            [sg.Column(
                layout=[
                    [sg.Canvas(key='fig_cv',
                            # it's important that you set this size
                            size=(400 * 2, 400)
                            )]
                ],
                background_color='white',pad=(0, 0)
            )],
            [sg.Push(),sg.B('Glint',key="-GLINT-"),sg.B('Non-glint',key="-NON_GLINT-"),\
                sg.B('Next Image',key="-NEXT_IMAGE_SGC-"),sg.B('Reset',key="-RESET_SGC-"),sg.B('Save',key="-SAVE_SGC-"),sg.Push()]]

    return sg.Window('Sunglint correction', layout,grab_anywhere=False,resizable=True,finalize=True,element_justification='center')

def get_canvas_window(g_size,canvas_scale):
    """
    plot stitch images on tkinter canvas
    g_size (tuple of integers): size of the graph in pixels
    """
    
    g_size_bigger = (int(canvas_scale*g_size[0]),int(canvas_scale*g_size[1]))

    col = [
        [sg.B('',image_data=move,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-MOVE-',tooltip='Move object')], #no default bool cus issa button!
        [sg.B('',image_data=moveall,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-MOVEALL-',tooltip='Move all objects')], #no default bool cus issa button!
        [sg.B('',image_data=erase,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ERASE-',tooltip='Erase object')], #no default bool cus issa button!
        [sg.B('',image_data=clear,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-CLEAR-',tooltip='clear all graphics')], #no default bool cus issa button!
        [sg.B('',image_data=bring_backward,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-BACK-',tooltip='Send backwards')], #no default bool cus issa button!
        [sg.B('',image_data=bring_forward,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-FRONT-',tooltip='Send forward')], #no default bool cus issa button!
        [sg.B('',image_data=background,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='Change background',tooltip='Change background colour')],
        [sg.B('',image_data=change_compass,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-CHANGE_COMPASS-',tooltip='Change compass graphic')],
        [sg.B('',image_data=add_text,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ADD_TEXT-',tooltip='Add text')],
        [sg.B('',image_data=scale_legend,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ADD_SCALE-',tooltip='Add scale')],
        [sg.B('',image_data=add_wql,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ADD_WQL-',tooltip='Add water quality points')],
        [sg.B('',image_data=add_spectral,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ADD_PREDICTED-',tooltip='Add predicted image')]
        ]

    layout = [   
            [sg.Col([[sg.Button('', image_data=sg.red_x,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='Exit',tooltip='Exit',pad=((0,20),(0,0)))]]),            
            sg.B('',image_data=add_spectral,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-ADD_SPECTRAL-',tooltip='Add extracted spectral graph'),
            sg.B('',image_data=add_image,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='Add Image',tooltip='Add Image'),
            sg.DropDown(values=[],size=(35,1),key='-GEOREFERENCED_IMAGE_LIST-'),
            sg.B('',image_data=save,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-SAVE-',tooltip='Save graphic'),
            ],
            [sg.Col(col, key='-COL-'),
            sg.Graph(
            canvas_size=g_size,
            graph_bottom_left=(0, 0),
            graph_top_right=g_size_bigger,
            key="-GRAPH-",
            enable_events=True,
            background_color='lightblue',
            drag_submits=True,
            right_click_menu=[[],['Erase item',]]
            ),
            sg.Col([[sg.Image(key='-SPECTRAL_CANVAS-')]])
            ],
        [sg.Text(key='-INFO-', size=(40, 1))]
        ]
    return sg.Window("Plot georeferenced images", layout, no_titlebar=True,grab_anywhere=True,finalize=True)

#------------------------------EVENT HANDLING------------------------------
main_window, table_window, hyperspectral_window, canvas_window, correction_canvas_window, sgc_canvas_window = get_main_window(), None, None, None, None, None
#------button states of main_window-----
main_window['Generate hyperspectral images'].update(disabled=True)
main_window['Plot spectral curve'].update(disabled=True)
main_window['View water quality file'].update(disabled=True)
main_window['View georeferenced images'].update(disabled=True)

options = {'-ERASE-':{'bool':False,'img_false':erase,'img_true':erase1},
            '-CLEAR-':{'bool':False,'img_false':clear,'img_true':clear1},
            '-FRONT-':{'bool':False,'img_false':bring_forward,'img_true':bring_forward1},
            '-BACK-':{'bool':False,'img_false':bring_backward,'img_true':bring_backward1},
            '-MOVEALL-':{'bool':False,'img_false':moveall,'img_true':moveall1},
            # '-RESIZE-':{'bool':False,'img_false':resize,'img_true':resize1},
            '-MOVE-':{'bool':False,'img_false':move,'img_true':move1},
            '-CHANGE_COMPASS-':{'bool':False,'img_false':change_compass,'img_true':change_compass1}
            } #use for buttons to indicate boolean
# options = ['-ERASE-','-CLEAR-','-FRONT-','-BACK-','-MOVEALL-','-MOVE-']
background_dict = {0:'lightblue',1:'white',2:'grey',3:'lightgrey'}
background_len = len(background_dict.keys())
background_counter = 0
add_spectral_bool = True
add_wql_bool = True
add_scale = True
add_predicted = True
#----compass----
compass_list = [north_arrow,north_arrow1,north_arrow2,north_arrow3,north_arrow4,north_arrow5,north_arrow6]
compass_offset = 0
corrected_indices = None
tss_lat = tss_lon = tss_measurements = None
text_canvas_ID = []
img_counter = 0
lock = "glint"
#----The Event Loop----
while True:
    # event,values = window.read()
    window, event, values = sg.read_all_windows()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == table_window:# if closing win 2, mark as closed
            table_window = None
            print('table window closed')
        elif window == hyperspectral_window:
            hyperspectral_window = None
            print('hyperspectral window closed')
        elif window == canvas_window:
            canvas_window = None
            print('canvas window closed')
        elif window == correction_canvas_window:
            correction_canvas_window = None
            print('correction canvas window closed')
        elif window == sgc_canvas_window:
            sgc_canvas_window = None
            print('sgc canvas window closed')
        elif window == main_window:# if closing win 1, exit program
            print('main window closed')
            break
    
    if event == "-CONFIG_FILEPATH-": #load config file
        config_fp = sg.popup_get_file('Config file to open')
        try:
            with open(config_fp) as cf:
                config_file = json.load(cf)
            
            try:
                config_file_events = {k:v for k,v in config_file.items() if '-' in k}
                print('config file:{}'.format(config_file_events))
                for k,v in config_file_events.items():
                    window[k].update(v)
            except Exception as E:
                sg.popup(f'{E}\nWindow elements not updated',title='Error')
                pass
            

        except Exception as E:
            sg.popup(f'{E}\nConfig file not found: {config_fp}',title='File not found')
            pass
        
        try:
            corrected_indices = config_file['corrected_indices']
            sg.popup(f"corrected_indices loaded: {corrected_indices}")
        except:
            corrected_indices = None
        print(f"Corrected_indices: {corrected_indices}")

        try:
            with open(values['-GPS_INDEX_TXT-'], "r") as f:
                gps_indices = f.readlines()
            gps_indices = [int(i.replace('\n','')) for i in gps_indices]
            gps_indices.sort()
            window['-GPS_INDEX-'].update(gps_indices)
            window['-LINE_START-'].update(values=[i for i in range(len(gps_indices)//2)])
            window['-LINE_END-'].update(values=[i for i in range(len(gps_indices)//2)])
            window['-NUMBER_OF_LINES-'].update('Flight lines range: 0 - {} lines'.format(len(gps_indices)//2 - 1))
        except Exception as E:
            sg.popup('{}\nGPS index file not found: {}'.format(E,values['-GPS_INDEX_TXT-']),title='File not found')
            pass
        

    if event == '-MASK_CHECKBOX-' and values['-MASK_CHECKBOX-'] is True:
        xgb_seg_fp = sg.popup_get_file('File to open containing segmentation model')
        
        try:
            xgb_seg_model = load_xgb_segmentation_model(xgb_seg_fp)
            sg.popup("Segmentation model successfully loaded",title="Model loaded")
        except Exception as E:
            sg.popup(f'{E}\n Model cannot be loaded',title="Error")
            window['-MASK_CHECKBOX-'].update(value = False)
            pass

    if event == 'Launch to select region':
        if values['-IMAGE_FOLDER_FILEPATH-'] == '':
            sg.popup(f"Image folder path is missing! Input it to continue.",title="Warning")
        else:
            try:
                draw_plot(values['-IMAGE_FOLDER_FILEPATH-'])
                if values['-PROCESSED_IMAGES-'] == '':
                    sg.popup(f"gps_index_{values['-PREFIX-']}.txt file created in folder: {values['-IMAGE_FOLDER_FILEPATH-']}!")
                    gps_fp = join(values['-IMAGE_FOLDER_FILEPATH-'],"gps_index_{}.txt".format(values['-PREFIX-']))
                else:
                    sg.popup(f"gps_index_{values['-PREFIX-']}.txt file created in folder: {values['-PROCESSED_IMAGES-']}!")
                    gps_fp = join(values['-PROCESSED_IMAGES-'],"gps_index_{}.txt".format(values['-PREFIX-']))
                with open(gps_fp, "r") as f:
                    gps_indices = f.readlines()
                gps_indices = [int(i.replace('\n','')) for i in gps_indices]
                gps_indices.sort() 
                window['-GPS_INDEX-'].update(gps_indices)
                if len(gps_indices) % 2 == 0: #even number of gps indices 
                    window['-NUMBER_OF_LINES-'].update('Flight lines range: 0 - {} lines'.format(len(gps_indices)//2 - 1))
                    #plot gps graph
                    gps_df = import_gps(values['-IMAGE_FOLDER_FILEPATH-'])
                    unique_gps_df = get_unique_df(gps_df)
                    test_gps_index = gps_indices
                    indexes_list = gps_to_image_indices(unique_gps_df,values['-IMAGE_FOLDER_FILEPATH-'],test_gps_index,int(values['-SLIDER-']))
                    reverse_boolean_list = rev_boolean_list(unique_gps_df,test_gps_index)
                    draw_gps(test_gps_index,unique_gps_df) #plot GPS graph
                else:
                    sg.popup(f"List of GPS indices is not an even number!\nNumber of GPS indices is {len(gps_indices)}.\nEach flight line must have a start and end index!\nPlease re-select the points using the launcher.",title="Warning")
            except Exception as E:
                print(f'** Error {E} **')
                pass
    elif event == '-GPS_INDEX_TXT-':
        if values['-IMAGE_FOLDER_FILEPATH-'] == '':
            sg.popup(f"Image folder path is missing! Input it to continue.",title="Warning")
        else:
            try:
                with open(values['-GPS_INDEX_TXT-'], "r") as f:
                    gps_indices = f.readlines()
                gps_indices = [int(i.replace('\n','')) for i in gps_indices]
                gps_indices.sort()
                # unique_gps_df,test_gps_index,indexes_list,reverse_boolean_list = event_gps_input(gps_indices)
                window['-GPS_INDEX-'].update(gps_indices)
                if len(gps_indices) % 2 != 0: #even number of gps indices 
                    sg.popup(f"List of GPS indices is not an even number!\nNumber of GPS indices is {len(gps_indices)}.\nEach flight line must have a start and end index!\nPlease re-select the points using the launcher.",title="Warning")
                    pass
                else:
                    window['-LINE_START-'].update(values=[i for i in range(len(gps_indices)//2)])
                    window['-LINE_END-'].update(values=[i for i in range(len(gps_indices)//2)])
                    window['-NUMBER_OF_LINES-'].update('Flight lines range: 0 - {} lines'.format(len(gps_indices)//2 - 1))
            except Exception as E:
                sg.popup(f"Error {E}",title="Error")
                pass
    
    if event == '-WQL_CSV-':
        main_window['View water quality file'].update(disabled=False)
        try:
            tss_df = pd.read_csv(values['-WQL_CSV-'],engine='python')   
        except Exception as E:
            sg.popup(f'{E}',title='Error in csv file!')
            pass
        
        try:
            df_columns = tss_df.columns
            window['-QUERY_VARIABLES-'].update(values=tuple(df_columns))
            window['-LAT_COL-'].update(values=tuple(df_columns))
            window['-LON_COL-'].update(values=tuple(df_columns))
            window['-WQL_COL-'].update(values=tuple(df_columns))
        except Exception as E:
            sg.popup(f'{E}.\nInvalid csv file input!',title='Error')
            pass

    elif event == '-QUERY_VARIABLES-':
        query_variable = '`' + values['-QUERY_VARIABLES-'] + '`'
        window['-QUERY_DF-'].update(values['-QUERY_DF-']+query_variable)

    elif event == 'View water quality file' and not table_window:
        
        try:
            if values['-QUERY_DF-'] != '':
                tss_df = tss_df.query(values['-QUERY_DF-'],inplace=False)
                data_table = tss_df.values.tolist()
                
            else:
                try:
                    tss_df = pd.read_csv(values['-WQL_CSV-'],engine='python')
                    data_table = tss_df.values.tolist()
                except Exception as E:
                    sg.popup(f'{E}',title='Error in csv file!')
                    pass
        except Exception as E:
            sg.popup(f'{E}',title='Query error!')
            pass

        header_list = df_columns.tolist()
        table_window = get_table_window(data_table,header_list)

    if event == '-MODEL_FILEPATH-':
        if values['-MODEL_FILEPATH-'].endswith('.model') == False and values['-MODEL_FILEPATH-'].endswith('.json') == False:
            sg.popup('Enter a valid model file in .model or .json format',title="Warning")
            pass
        if values['-PREDICT_CHECKBOX-'] == False:
            sg.popup('Check the box above to enable prediction of TSS map, otherwise prediction will not take place.',title="Warning")
    
    
    if event == 'Process':
        start_time = time.time()
        #enter the list of required info here
        required_info = {'image folder path':values['-IMAGE_FOLDER_FILEPATH-'],'spectrometer folder path':values['-SPECTRO_FILEPATH-'],'height':values['-HEIGHT-'],'gps indices file path':values['-GPS_INDEX_TXT-']}
        
        if '' in required_info.values():
            missing_info_list = [k for k,v in required_info.items() if v == '']
            sg.popup('These elements are missing: {}\nInput these information, then click on PROCESS to continue!'.format(missing_info_list),title='Missing required elements!')
            pass
        else:
            #----GPS indices info---
            if len(gps_indices) % 2 != 0: #even number of gps indices 
                sg.popup(f"List of GPS indices is not an even number!\nNumber of GPS indices is {len(gps_indices)}.\nEach flight line must have a start and end index!\nPlease re-select the points using the launcher.",title="Warning")
                pass
            elif len(gps_indices) % 2 == 0: #even number of gps indices 
                #plot gps graph
                gps_df = import_gps(values['-IMAGE_FOLDER_FILEPATH-'])
                unique_gps_df = get_unique_df(gps_df)
                test_gps_index = gps_indices
                indexes_list = gps_to_image_indices(unique_gps_df,values['-IMAGE_FOLDER_FILEPATH-'],test_gps_index,int(values['-SLIDER-']))
                reverse_boolean_list = rev_boolean_list(unique_gps_df,test_gps_index)
                draw_gps(test_gps_index,unique_gps_df) #plot GPS graph

            else:
                sg.popup(f"GPS indices error. GPS indices likely not in the correct format",title="Warning")
                pass

            if values['-LINE_START-']=='' and values['-LINE_END-']=='':
                sg.popup("All lines indicated by gps indices will be processed",title="Alert")
                line_start = 0
                line_end = len(gps_indices)//2-1
            elif values['-LINE_END-'] == '':
                line_start = values['-LINE_START-']
                line_end = len(gps_indices)//2-1
            elif values['-LINE_START-']!='' and values['-LINE_END-']!='':
                line_start = values['-LINE_START-']
                line_end = values['-LINE_END-']
            else:
                sg.popup("Please ensure both line start and line end fields are entered!",title="Error")
                pass

            if values['-PROCESSED_IMAGES-'] == '':
                fp_store = values['-IMAGE_FOLDER_FILEPATH-']
            else:
                fp_store = values['-PROCESSED_IMAGES-']

            r,g,b = 38,23,15
            if values['-R-'] != '':
                R = re.search('\(.*\)', values['-R-'])
                r = int(R.group(0).lower().replace(')','').replace('(band',''))
            if values['-G-'] != '':
                G = re.search('\(.*\)', values['-G-'])
                g = int(G.group(0).lower().replace(')','').replace('(band',''))
            if values['-B-'] != '':
                B = re.search('\(.*\)', values['-B-'])
                b = int(B.group(0).lower().replace(')','').replace('(band',''))

            plot_flight_camera_attributes(unique_gps_df,int(values['-HEIGHT-']),test_gps_index=test_gps_index)

            if values['-PREDICT_CHECKBOX-'] == True and (values['-MODEL_FILEPATH-'].endswith('.model') == True or values['-MODEL_FILEPATH-'].endswith('.json') == True):
                model_type = "XGBoost"
                if values['-MODEL_PREDICTORS-'] != '':
                    covariates_index = values['-MODEL_PREDICTORS-'].replace(' ','').split(',')
                    covariates_index_list = []
                    for i in covariates_index:
                        if ':' in i:
                            c_start, c_end = i.split(':')
                            # print("c_start,c_end:",c_start, c_end)
                            try:
                                covariates_index_list = covariates_index_list + list(range(int(c_start),int(c_end)+1))
                            except Exception as E:
                                sg.popup("String contains letters!",title="Warning")
                                pass
                        else:
                            covariates_index_list.append(int(i))
                else:
                    covariates_index_list = list(range(len(bands_wavelengths())))
            #---store config file---
            config_file = {'-PROCESSED_IMAGES-':fp_store,'-PREFIX-':values['-PREFIX-'],'-IMAGE_FOLDER_FILEPATH-':values['-IMAGE_FOLDER_FILEPATH-'],\
                '-SPECTRO_FILEPATH-':values['-SPECTRO_FILEPATH-'],\
                '-HEIGHT-':int(values['-HEIGHT-']),'-GPS_INDEX_TXT-':values['-GPS_INDEX_TXT-'],'-SLIDER-':values['-SLIDER-'],\
                '-LINE_START-':line_start,'-LINE_END-':line_end,'rgb_bands':[r,g,b],\
                '-MASK_CHECKBOX-':values['-MASK_CHECKBOX-'],'-CLASSIFY_CHECKBOX-':values['-CLASSIFY_CHECKBOX-'],'-NOISE_CHECKBOX-':values['-NOISE_CHECKBOX-'],\
                '-SUNGLINT_CHECKBOX-':values['-SUNGLINT_CHECKBOX-'],'-PREDICT_CHECKBOX-':values['-PREDICT_CHECKBOX-'],\
                'corrected_indices':corrected_indices}

            if values['-PREDICT_CHECKBOX-'] == True and (values['-MODEL_FILEPATH-'].endswith('.model') == True or values['-MODEL_FILEPATH-'].endswith('.json') == True):
                model_type = "XGBoost"
                prediction_parameters = ['-MODEL_PREDICTORS-','-MODEL_PREDICTORS-','-RADIOMETRIC_CHECKBOX-','-DOWNSAMPLE_SLIDER-','-PREDICT_CHECKBOX-','-CLASSIFY_CHECKBOX-','-MODEL_FILEPATH-']
                for p in prediction_parameters:
                    config_file[p] = values[p]
                config_file['model_type'] = model_type
            if values['-WQL_CSV-'] is not None:
                for p in ['-WQL_CSV-','-LAT_COL-','-LON_COL-','-WQL_COL-']:
                    config_file[p] = values[p]
            #---store config file---
            if 'xgb_seg_model' not in globals() and values['-MASK_CHECKBOX-'] is True:
                xgb_seg_fp = sg.popup_get_file('Segmentation model not loaded yet. File to open containing segmentation model')
                try:
                    xgb_seg_model = load_xgb_segmentation_model(xgb_seg_fp)
                    sg.popup("Segmentation model successfully loaded",title="Model loaded")
                except Exception as E:
                    sg.popup(f'{E}\nModel cannot be loaded',title="Error")
                    window['-MASK_CHECKBOX-'].update(value = False)
                    values['-MASK_CHECKBOX-'] = False
                    pass

            with open(join(fp_store,'config_file_{}.txt'.format(values['-PREFIX-'])),'w') as cf:
                json.dump(config_file,cf)

            required_wql_info = {'wql csv file':values['-WQL_CSV-'],'latitude column index':values['-LAT_COL-'],'longitude column index':values['-LON_COL-'],'wql column index':values['-WQL_COL-']}
            number_wql_fields = len(required_wql_info.keys())
            missing_wql_inputs = [k for k,v in required_wql_info.items() if v == ''] #missing inputs
            if len(missing_wql_inputs) >= 1 and len(missing_wql_inputs) < len(required_wql_info.keys()):
                sg.popup(f'These wql elements are missing: {missing_wql_inputs}!\nDo you wish to extract the spectral information? Enter all the fields, otherwise clear all the fields in this section!\nThen click on PROCESS to continue.',title='Missing wql inputs')
            elif len(missing_wql_inputs) == len(required_wql_info.keys()):
                sg.popup('Water quality fields are empty. No spectral information will be extracted.')
                #stitch images only but don't extract spectral information
            else:
                sg.popup('Spectral information will be extracted from water quality points.')
                try:
                    tss_lat = tss_df[values['-LAT_COL-']].tolist()
                    tss_lon = tss_df[values['-LON_COL-']].tolist()
                    tss_measurements = tss_df[values['-WQL_COL-']].tolist()
                except Exception as E:
                    sg.popup(f'{E}\n Column name not found or has weird symbols. Remove symbols and continue.',title='Column name not found')
                    tss_lat = tss_lon = tss_measurements = None
            
            if corrected_indices is not None:
                indexes_list = corrected_indices #overwrite indexes list if corrected_indices exists

            for line_number in range(line_start, line_end+1):
                sg.one_line_progress_meter('Stitching RGB images in progress', line_number, line_end, f"Processing image line {line_number}...", orientation='h')
                start_i,end_i = indexes_list[line_number]#these are image_indices
                test_stitch_class = StitchHyperspectral(fp_store,values['-PREFIX-'],values['-IMAGE_FOLDER_FILEPATH-'],values['-SPECTRO_FILEPATH-'],\
                    int(values['-HEIGHT-']),line_number,start_i,end_i,\
                    test_gps_index, unique_gps_df, destriping = values['-NOISE_CHECKBOX-'],\
                    reverse=reverse_boolean_list[line_number])
                test_stitch_class.view_pseudo_colour(r,g,b) #outputs rgb

            #-----------sunglint correction requires processing of all rgb images first------------
            if values['-SUNGLINT_CHECKBOX-'] is True:
                #if sgc json file has been saved, next time dont need to select glint areas again, just click proceed
                sgc_canvas_window = get_sgc_canvas()
                fig, ax = plt.subplots()
                DPI = fig.get_dpi()
                # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
                fig.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
            # -------------------------------
                rgb_fp = [join(fp_store,f) for f in listdir(fp_store) if ("Geotransformed" not in f and "predicted" not in f) and (f.endswith(".tif"))]
                n_imges = len(rgb_fp)
                #draw img
                current_fp = rgb_fp[img_counter%n_imges]
                img = Image.open(current_fp)
                img_line = int(rgb_fp[img_counter%n_imges].split('line_')[1][:2])
                ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
                im = ax.imshow(img)
                #draw lines
                line_glint, = ax.plot(0,0,"o",c="r")
                line_nonglint, = ax.plot(0,0,"o",c="purple")
                # Create a Rectangle patch
                rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
                r_glint = ax.add_patch(rect_glint)
                rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
                r_nonglint = ax.add_patch(rect_nonglint)
                # linebuilder = draw_figure_w_toolbar(sgc_canvas_window['fig_cv'].TKCanvas,fig,sgc_canvas_window['controls_cv'].TKCanvas,"test")
                figure_canvas_agg = draw_figure_w_toolbar(canvas=sgc_canvas_window['fig_cv'].TKCanvas,fig=fig, canvas_toolbar=sgc_canvas_window['controls_cv'].TKCanvas)
                linebuilder = LineBuilder(xs_glint=[],ys_glint=[],xs_nonglint=[],ys_nonglint=[],\
                    line_glint=line_glint,line_nonglint=line_nonglint,r_glint=r_glint,r_nonglint=r_nonglint,\
                    img_line_glint=current_fp,img_line_nonglint=current_fp,\
                    img_bbox_glint=None,img_bbox_nonglint=None,\
                    canvas=figure_canvas_agg,lock=lock,current_fp = current_fp)

            else: #no sunglint correction, proceed with the remaining process
                print("No sunglint correction...")
                for line_number in range(line_start, line_end+1):
                    sg.one_line_progress_meter('Processing in progress', line_number, line_end, f"Processing image line {line_number}...", orientation='h')
                    start_i,end_i = indexes_list[line_number]#these are image_indices
                    test_stitch_class = StitchHyperspectral(fp_store,values['-PREFIX-'],values['-IMAGE_FOLDER_FILEPATH-'],values['-SPECTRO_FILEPATH-'],\
                        int(values['-HEIGHT-']),line_number,start_i,end_i,\
                        test_gps_index, unique_gps_df,reverse=reverse_boolean_list[line_number])
                    if values['-MASK_CHECKBOX-'] is True:
                        print("Performing segmentation...")
                        mask = test_stitch_class.get_mask(xgb_seg_model,type="XGBoost")
                    else:
                        print("Segmentation not conducted...")
                        mask = None
                    # test_stitch_class.view_pseudo_colour(r,g,b,"destriping_array.csv")
                    if (values['-PREDICT_CHECKBOX-'] == True and (values['-MODEL_FILEPATH-'].endswith('.model') == True or values['-MODEL_FILEPATH-'].endswith('.json') == True)) or tss_lat is not None:
                        #produce reflectance for prediction OR extraction of spectral information
                        sgc_reflectance = None
                        if values['-RADIOMETRIC_CHECKBOX-'] is True:
                            reflectance = test_stitch_class.get_stitched_reflectance()
                            print("Getting radiometrically corrected reflectances...")
                        else:
                            reflectance = test_stitch_class.get_stitched_uncorrected_reflectance()
                            print("Getting uncorrected reflectances...")

                    if values['-PREDICT_CHECKBOX-'] == True and (values['-MODEL_FILEPATH-'].endswith('.model') == True or values['-MODEL_FILEPATH-'].endswith('.json') == True):
                        #prediction
                        try: #4:54, #4:15,33:38,43:46,53:54
                            test_stitch_class.get_predicted_image(model=values['-MODEL_FILEPATH-'],\
                                covariates_index_list = covariates_index_list,\
                                model_type=model_type,\
                                # radiometric_correction=values['-RADIOMETRIC_CHECKBOX-'],\
                                scaling_factor=int(values['-DOWNSAMPLE_SLIDER-']),\
                                glint_corrected_reflectance=sgc_reflectance,reflectance=reflectance)
                            print("Generating prediction map...")
                        except Exception as E:
                            sg.popup(f'{E}',title="Warning!")
                            print("No prediction...")
                            pass
                    if tss_lat is not None:
                        #extraction of spectral information
                        test_stitch_class.get_reflectance_from_GPS(tss_lat,tss_lon,tss_measurements,\
                            radius=2, mask = mask,\
                            glint_corrected_reflectance=sgc_reflectance,reflectance = reflectance)
                    gti = GeotransformImage(test_stitch_class,\
                        mask = mask, classify = values['-CLASSIFY_CHECKBOX-'],\
                        transform_predicted_image = values['-PREDICT_CHECKBOX-'],\
                        sunglint_correction=values['-SUNGLINT_CHECKBOX-'])
                    gti.geotransform_image()
                
                if tss_lat is not None:
                    test_stitch_class.preprocess_spectral_info(export_to_array=False)

                end_time = time.time()
                print('Stitching completed!')
                sg.popup('Stitching completed!\nTotal time taken: {:.1f} seconds'.format(end_time - start_time),title='Progress')
                
                try:
                    file_list = listdir(fp_store)
                except:
                    file_list = []
                fnames = [f for f in file_list if isfile(
                    join(fp_store, f)) and f.lower().endswith((".tif"))]
                fnames.sort()
                window['-IMAGE LIST-'].update(fnames)
            
    if event == "-PROCEED_PROCESS-":
        #if sgc json file has been saved, next time dont need to select glint areas again, just click proceed
        #-----close sunglint correction window---------
        sg.popup("Proceeding with the rest of processing...")
        if window == sgc_canvas_window:
            window.close()
            sgc_canvas_window = None
        #-------------load sgc json file---------------
        try:
            sunglint_json_fp = join(fp_store,'sunglint_correction_{}.txt'.format(config_file['-PREFIX-']))
            with open(sunglint_json_fp,"r") as cf:
                bbox = json.load(cf)
            
        except Exception as E:
            sg.popup(f"{E}",title="Error")
            sunglint_json_fp = None
            reflectance_glint = None
        try:
            line_glint = bbox['glint']['line']
            start_i,end_i = indexes_list[line_glint]
            test_stitch_class = StitchHyperspectral(fp_store,config_file['-PREFIX-'],config_file['-IMAGE_FOLDER_FILEPATH-'],config_file['-SPECTRO_FILEPATH-'],\
                int(config_file['-HEIGHT-']),line_glint,start_i,end_i,\
                test_gps_index, unique_gps_df,reverse=reverse_boolean_list[line_glint])

            reflectance_glint = test_stitch_class.get_stitched_reflectance() #radiometrically corrected reflectance only for the image line where glint bbox is drawn on
        except Exception as E:
            sg.popup(f"{E}",title="Error")
            sunglint_json_fp = None
            reflectance_glint = None
            
        #-----closed sunglint correction window and proceed with the remaining---------
        for line_number in range(line_start, line_end+1):
            sg.one_line_progress_meter('Processing in progress', line_number, line_end, f"Processing image line {line_number}...", orientation='h')
            start_i,end_i = indexes_list[line_number]#these are image_indices
            test_stitch_class = StitchHyperspectral(fp_store,config_file['-PREFIX-'],config_file['-IMAGE_FOLDER_FILEPATH-'],config_file['-SPECTRO_FILEPATH-'],\
                int(config_file['-HEIGHT-']),line_number,start_i,end_i,\
                test_gps_index, unique_gps_df,reverse=reverse_boolean_list[line_number])
            if config_file['-SUNGLINT_CHECKBOX-'] is True and sunglint_json_fp is not None and reflectance_glint is not None:
                sgc = SunglintCorrection(test_stitch_class,sunglint_json_fp,reflectance_glint)
                sgc.sunglint_correction_rgb()
                print("Performing sunglint correction on RGB images...")
            # if values['-MASK_CHECKBOX-'] is True:
            if config_file['-MASK_CHECKBOX-'] is True:
                print("Performing segmentation...")
                mask = test_stitch_class.get_mask(xgb_seg_model,type="XGBoost")
            else:
                print("Segmentation not conducted...")
                mask = None
            if (config_file['-PREDICT_CHECKBOX-'] == True and (config_file['-MODEL_FILEPATH-'].endswith('.model') == True or config_file['-MODEL_FILEPATH-'].endswith('.json') == True)) or tss_lat is not None:
                if config_file['-SUNGLINT_CHECKBOX-'] is True and sunglint_json_fp is not None and reflectance_glint is not None:
                    sgc_reflectance = sgc.sunglint_correction_reflectance()
                    reflectance = None
                    print("Performing sunglint correction on hyperspectral reflectances...")
                else:
                    print("No sunglint correction...")
                    config_file['-SUNGLINT_CHECKBOX-'] = False
                    sgc_reflectance = None
                    if config_file['-RADIOMETRIC_CHECKBOX-'] is True:
                        reflectance = test_stitch_class.get_stitched_reflectance()
                        print("Getting radiometrically corrected reflectances...")
                    else:
                        reflectance = test_stitch_class.get_stitched_uncorrected_reflectance()
                        print("Getting uncorrected reflectances...")
            else:
                print("No prediction or extraction of spectral information or sunglint correction...")
                #no prediction, no extraction of spectral information. Then no need for sunglint correction reflectance
                reflectance = None
                sgc_reflectance = None
                tss_lat = None
                config_file['-PREDICT_CHECKBOX-'] = False
                config_file['-SUNGLINT_CHECKBOX-'] = False

            if config_file['-PREDICT_CHECKBOX-'] == True and (config_file['-MODEL_FILEPATH-'].endswith('.model') == True or config_file['-MODEL_FILEPATH-'].endswith('.json') == True):
                try: #4:54, #4:15,33:38,43:46,53:54
                    test_stitch_class.get_predicted_image(model=config_file['-MODEL_FILEPATH-'],\
                        covariates_index_list = covariates_index_list,\
                        model_type=model_type,\
                        scaling_factor=int(config_file['-DOWNSAMPLE_SLIDER-']),\
                        glint_corrected_reflectance=sgc_reflectance,reflectance=reflectance)
                    print("Generating prediction map...")
                except Exception as E:
                    sg.popup(f'{E}',title="Warning!")
                    print("No prediction...")
                    pass
            if tss_lat is not None:
                test_stitch_class.get_reflectance_from_GPS(tss_lat,tss_lon,tss_measurements,\
                    radius=2, mask = mask,\
                    glint_corrected_reflectance=sgc_reflectance,reflectance = reflectance)
                print("Extracting spectral information...")
            gti = GeotransformImage(test_stitch_class,\
                mask = mask, classify = config_file['-CLASSIFY_CHECKBOX-'],\
                transform_predicted_image = config_file['-PREDICT_CHECKBOX-'],\
                sunglint_correction=config_file['-SUNGLINT_CHECKBOX-'])
            gti.geotransform_image()
        
        if tss_lat is not None:
            test_stitch_class.preprocess_spectral_info(export_to_array=False)

        end_time = time.time()
        print('Stitching completed!')
        sg.popup('Stitching completed!\nTotal time taken: {:.1f} seconds'.format(end_time - start_time),title='Progress')
        
        try:
            file_list = listdir(fp_store)
        except:
            file_list = []
        fnames = [f for f in file_list if isfile(
            join(fp_store, f)) and f.lower().endswith((".tif"))]
        fnames.sort()
        main_window['-IMAGE LIST-'].update(fnames)
    
    
    if event == '-BROWSE_IMAGES-':
        try:
            file_list = listdir(values['-BROWSE_IMAGES-'])
        except:
            file_list = []
        fnames = [f for f in file_list if isfile(
            join(values['-BROWSE_IMAGES-'], f)) and f.lower().endswith((".tif"))]
        fnames.sort()
        window['-IMAGE LIST-'].update(fnames)
        fp_store = values['-BROWSE_IMAGES-']

        try:
            spectral_csv_files = [f for f in listdir(join(fp_store,'Extracted_Spectral_Information')) if f.endswith('TSS_spectral_info.csv')]
        except Exception as E:
            sg.popup(f'{E}',title='Error')
            spectral_csv_files = None
            pass

        if spectral_csv_files is not None:
            spectral_csv_files.sort()
            window['-PROCESSED_WQL_CSV-'].update(values=spectral_csv_files)
            window['Plot spectral curve'].update(disabled=False)

        prefix_list = list(set([re.sub(r'_Geotransformed_rgb_image_line.*?$', '', f) for f in listdir(fp_store) if 'Geotransformed_rgb_image_line' in f]))
        geotransformed_dict = {k:[] for k in prefix_list} #where keys are the prefix, and values are the list of geotransformed filepath names
        for f in listdir(fp_store):
            if 'Geotransformed_rgb_image_line' in f:
                geotransformed_prefix = re.sub(r'_Geotransformed_rgb_image_line.*?$', '', f)
                geotransformed_dict[geotransformed_prefix].append(f)
        prefix_list.sort()
        window['-PROCESSED_GEOTRANSFORMED_IMAGES-'].update(values=prefix_list)

    elif event == '-PROCESSED_GEOTRANSFORMED_IMAGES-':
        window['View georeferenced images'].update(disabled=False)
        geotransformed_prefix = values['-PROCESSED_GEOTRANSFORMED_IMAGES-']
    elif event == 'Plot spectral curve':
        tss_df_fp = join(fp_store,'Extracted_Spectral_Information',values['-PROCESSED_WQL_CSV-'])
        
        try:
            tss_full_df = pd.read_csv(tss_df_fp)
            
        except Exception as E:
            sg.popup('{} cannot be found! Please check if the csv file is found inside the folder \'Extracted_Spectral_Information'.format(tss_df_fp))
            pass
        
        try:
            plot_TSS_conc(tss_full_df) #---plot TSS graph, include map of stitched image---
        except Exception as E:
            sg.popup(f'{E}')
            pass

    elif event == '-IMAGE LIST-': # A file was chosen from the listbox
        filename = join(fp_store, values['-IMAGE LIST-'][0])
        window['-TOUT-'].update(filename)
        if 'predicted_image_line' in values['-IMAGE LIST-'][0]:
            fig = plot_predicted_image(filename)
            image = figure_to_image(fig)
            image = convert_to_bytes(image, (500,500))
            window['-IMAGE-'].update(data=image)
        else:
            try:            
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(300,300)))
            except Exception as E:
                print(f'** Error {E} **')
                pass
            if 'Geotransformed' not in values['-IMAGE LIST-'][0] and 'rgb' in values['-IMAGE LIST-'][0]:
                window['Generate hyperspectral images'].update(disabled=False)
            else:
                window['Generate hyperspectral images'].update(disabled=True)

    if event == 'Generate hyperspectral images':
        rgb_filename = values['-IMAGE LIST-'][0]#.replace('rgb_image_line_','').replace('.tif','')
        prefix = re.match("(.*?)rgb_image_line",rgb_filename).group().replace('_rgb_image_line','')
        rgb_filename = re.sub(r'^.*?line_', '', rgb_filename).replace('.tif','')
        line_number, start_i, end_i = rgb_filename.split('_')
        try:
            with open(join(fp_store,'config_file_{}.txt'.format(prefix))) as cf:
                config_file = json.load(cf)
        except Exception as E:
            sg.popup(f'{E}\nconfig_file_{prefix}.txt not found in the processed images folder. Did you shift the images out of its original processed folder? Please locate the original config_file_{prefix}.txt and shift it to the selected browsed image folder',title='Config file not found!')
            pass
        
        sg.popup('Generating hyperspectral images for {}!'.format(values['-IMAGE LIST-'][0]),title='Progress')
        try:
            gps_df = import_gps(config_file['-IMAGE_FOLDER_FILEPATH-'])
        except Exception as E:
            sg.popup('{}\nAre the images stored in {}? Raw images could not be found in this folder. Otherwise, browse folder under Image folder path to indicate where the corresponding image folder is stored'.format(E,config_file['-IMAGE_FOLDER_FILEPATH-']))
            pass
        unique_gps_df = get_unique_df(gps_df)
        # gps_fp = join(fp_store,"gps_index_{}.txt".format(config_file['prefix'])) #uses prefix from config file because gps_index file may have another prefix that can also be used to process a set of images with diff prefix in the other instances
        gps_fp = config_file['-GPS_INDEX_TXT-']
        try:
            with open(gps_fp, "r") as f:
                gps_indices = f.readlines()
        except Exception as E:
            sg.popup(f'Please check if {gps_fp} exist in the folder where processed images are stored')
            pass

        if not exists(config_file['-SPECTRO_FILEPATH-']):
            sg.popup('Spectrometer folder is not found in {}!\nOtherwise, browse folder under Spectrometer folder path to indicate where the corresponding spectrometer folder is stored'.format(E,config_file['-SPECTRO_FILEPATH-']))
            pass

        gps_indices = [int(i.replace('\n','')) for i in gps_indices]
        gps_indices.sort()
        test_gps_index = gps_indices
        indexes_list = gps_to_image_indices(unique_gps_df,config_file['-IMAGE_FOLDER_FILEPATH-'],test_gps_index,int(values['-SLIDER-']))
        reverse_boolean_list = rev_boolean_list(unique_gps_df,test_gps_index)

        test_stitch_class = StitchHyperspectral(fp_store,prefix,config_file['-IMAGE_FOLDER_FILEPATH-'],config_file['-SPECTRO_FILEPATH-'],\
            config_file['-HEIGHT-'],int(line_number),int(start_i),int(end_i),\
            test_gps_index, unique_gps_df,reverse=reverse_boolean_list[int(line_number)])
        test_stitch_class.generate_hyperspectral_images() #outputs individual band images inside the hyperspectral 
        sg.popup('Hyperspectral images generated!',title='Progress')

    if event == 'Select hyperspectral bands':
        hyperspectral_window = get_hyperspectral_window()

    elif event == '-BROWSE_HYPERSPECTRAL_IMAGES-':
        try:
            if 'Hyperspectral' not in values['-BROWSE_HYPERSPECTRAL_IMAGES-']:
                sg.popup('Hyperspectral folder is not selected!',title="Warning")
                pass
            # file_list = listdir(folder_processed_images)
            else:
                image_list = listdir(values['-BROWSE_HYPERSPECTRAL_IMAGES-'])
        except:
            image_list = []

        unique_fname = {}
        for f in image_list:
            prefix = re.sub(r'band.*?$', '', f)#.replace('.tif','').split('_')
            postfix = re.sub(r'^.*?image_line_', 'image_line_', f).replace('.tif','')
            filename = prefix+postfix
            unique_fname[filename] = {'prefix':prefix,'postfix':postfix}

        window['-HYPERSPECTRAL_IMAGE_LIST-'].update(list(unique_fname.keys()))
    
    elif event == '-HYPERSPECTRAL_IMAGE_LIST-': # A file was chosen from the listbox
        image_dict = unique_fname[values['-HYPERSPECTRAL_IMAGE_LIST-'][0]]
        r = join(values['-BROWSE_HYPERSPECTRAL_IMAGES-'],image_dict['prefix']+'band38_'+image_dict['postfix']+'.tif')
        g = join(values['-BROWSE_HYPERSPECTRAL_IMAGES-'],image_dict['prefix']+'band23_'+image_dict['postfix']+'.tif')
        b = join(values['-BROWSE_HYPERSPECTRAL_IMAGES-'],image_dict['prefix']+'band15_'+image_dict['postfix']+'.tif')
        try:
            r = Image.open(r)
            g = Image.open(g)
            b = Image.open(b)
        except Exception as E:
            print(f'** Error {E} **')
            pass
        stacked = np.dstack((r,g,b))
        img = Image.fromarray(stacked, 'RGB')
        window['-HYPERSPECTRAL_IMAGE-'].update(data=convert_to_bytes1(img, resize=(500,500)))

    elif event == '-RED_BAND-' or event == '-GREEN_BAND-' or event == '-BLUE_BAND-':
        if event == '-RED_BAND-':
            band = values['-RED_BAND-']
        elif event == '-GREEN_BAND-':
            band = values['-GREEN_BAND-']
        else:
            band = values['-BLUE_BAND-']
        band = re.search('\(.*\)', band)
        band = band.group(0).lower().replace(' ','').replace(')','').replace('(','')
        band_fp = join(values['-BROWSE_HYPERSPECTRAL_IMAGES-'],image_dict['prefix']+band+'_'+image_dict['postfix']+'.tif')
        try:
            band_img = Image.open(band_fp)
        except Exception as E:
            print(f'** Error {E} **')
            pass
        if event == '-RED_BAND-':
            r = band_img
        elif event == '-GREEN_BAND-':
            g = band_img
        else:
            b = band_img
        stacked = np.dstack((r,g,b))
        img = Image.fromarray(stacked, 'RGB')
        window['-HYPERSPECTRAL_IMAGE-'].update(data=convert_to_bytes1(img, resize=(500,500)))    

    
    if event == 'View georeferenced images':
        geotransformed_prefix = values['-PROCESSED_GEOTRANSFORMED_IMAGES-']
        
        with open(join(fp_store,"config_file_{}.txt".format(geotransformed_prefix))) as cf:
            config_file = json.load(cf)

        scale = 20
        canvas_scale = 1.5
        compass_scale = 2

        vc = ViewCanvas(config_file,scale)
        img_resize_list,bytes_list, tss_full_df,wql_legend_content = vc.rgb_canvas()
        # print(vc.gps_indices)

        print(f'List of geotransformed files: {vc.rgb_fp_list}')
        print(f'List of predicted files: {vc.prediction_fp_list}')
        
        if img_resize_list[0].shape[0] > img_resize_list[0].shape[1]:
            g_size = (img_resize_list[0].shape[0]*canvas_scale,img_resize_list[0].shape[0]*canvas_scale)
        else:
            g_size = (img_resize_list[0].shape[1]*canvas_scale,img_resize_list[0].shape[1]*canvas_scale)

        canvas_window = get_canvas_window(g_size,compass_scale)
        canvas_window['-GEOREFERENCED_IMAGE_LIST-'].update(values=vc.rgb_fp_list)
        print(f'image_lines_list: {vc.rgb_fp_list}')
        graph = canvas_window["-GRAPH-"]  # type: sg.Graph
        #-----------------plot images---------------
        image_canvas_id = []
        for b in bytes_list:
            image_canvas = graph.draw_image(data=b, location=(0,int(img_resize_list[0].shape[0]*compass_scale))) #if doesnt work change to n_cols
            image_canvas_id.append(image_canvas)
        prev_image_coord, _ = graph.get_bounding_box(image_canvas)
        #plot compass after images
        n_compass = len(compass_list)
        compass_canvas = graph.draw_image(data=compass_list[compass_offset%n_compass], location=(0,int(img_resize_list[0].shape[0]*compass_scale)))
        prev_compass_coord, _ = graph.get_bounding_box(compass_canvas)

        #----------predicted images----------------
        if vc.prediction_fp_list is not None: 
            pred_img_resize_list,pred_bytes_list,tss_full_df,wql_legend_content = vc.prediction_canvas()
        else:
            pred_img_resize_list,pred_bytes_list = None,None # dont replace the df and wql_legend from rgb_canvas
            add_predicted = None

        #---wql info---
        if tss_full_df is None:
            sg.popup('{} cannot be found! Please check if the csv file is found inside the folder \'Extracted_Spectral_Information.\nWQL points and spectral info cannot be plotted!\nOr wql data is empty or only has 1 row'.format(vc.wql_csv))
            add_spectral_bool = None
            add_wql_bool = None
            pass

        #global variables
        global_vars = {'image_canvas_id':image_canvas_id,str(compass_canvas):compass_canvas}

        dragging = False
        start_point = end_point = prior_rect = None

    elif event in ('-MOVE-', '-MOVEALL-'):
        graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
    elif event is not None and not event.startswith('-GRAPH-') and window == canvas_window:
        graph.set_cursor(cursor='left_ptr')       # not yet released method... coming soon!

    elif event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
        graph.grab_anywhere_exclude()
        x, y = values["-GRAPH-"]
        if not dragging:
            start_point = (x, y)
            dragging = True
            drag_figures = graph.get_figures_at_location((x,y))
            lastxy = x, y
        else:
            end_point = (x, y)
        if prior_rect:
            graph.delete_figure(prior_rect)
        delta_x, delta_y = x - lastxy[0], y - lastxy[1]
        lastxy = x,y
        if None not in (start_point, end_point):
            if options['-MOVE-']['bool'] == True:
                #in global_vars (dict), keys with 'id' in the variable name all move together with figures
                move_group = [v for k,v in global_vars.items() if 'id' in k] #list of list of items to move together e.g img,scale_grid_id, wql points
                move_group = [item for sublist in move_group for item in sublist] #flatten list which combines all the ids
                move_ind_group = [v for k,v in global_vars.items() if 'id' not in k] #list of list of ind groups e.g. legend
                # print(f"Move_ind_group: {move_ind_group}\ndrag_figures: {drag_figures}")
                if any(item in move_group for item in drag_figures) == True:
                    for fig in move_group:
                        graph.move_figure(fig, delta_x, delta_y)
                        graph.update()
                for g in move_ind_group:
                    try:
                        if any(item in g for item in drag_figures) == True: #for moving legends etc
                            for fig in g:
                                graph.move_figure(fig, delta_x, delta_y)
                                graph.update()
                    except: #for moving text
                        if g in drag_figures:#if there is no list but only an individual element e.g. text
                            graph.move_figure(g, delta_x, delta_y)
                            graph.update()

                
            elif options['-ERASE-']['bool'] == True:
                for k in list(global_vars):
                    
                    try:
                        if any(item in global_vars[k] for item in drag_figures) == True:
                            for fig in global_vars[k]:
                                graph.delete_figure(fig)
                            del global_vars[k]
                    except:
                        if global_vars[k] in drag_figures == True: #if it's a single element (not in a list) e.g. text ID
                            graph.delete_figure(global_vars[k])
                            del global_vars[k]

                if 'image_canvas_id' not in list(global_vars):
                #then delete everything
                    for k in list(global_vars):
                        try: #delete grouped objects
                            for fig in global_vars[k]:
                                graph.delete_figure(fig)
                            del global_vars[k]
                        except: #delete individual elements
                            graph.delete_figure(global_vars[k]) 
                            del global_vars[k]
                    image_canvas_id = [] #create an empty list of images in case we want to add images later
                

            elif options['-CLEAR-']['bool'] == True:
                graph.erase()
                # for k in global_vars.keys():
                #     del global_vars[k]
                for k in list(global_vars):
                    del global_vars[k]
                image_canvas_id = []
                
            elif options['-MOVEALL-']['bool'] == True:
                graph.move(delta_x, delta_y)
            elif options['-FRONT-']['bool'] == True:
                for fig in drag_figures:
                    graph.bring_figure_to_front(fig)
            elif options['-BACK-']['bool'] == True:
                for fig in drag_figures:
                    graph.send_figure_to_back(fig)

        window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")
    elif event is not None and event.endswith('+UP') and window == canvas_window:  # The drawing has ended because mouse up
        window["-INFO-"].update(value=f"grabbed object from {start_point} to {end_point}")
        start_point, end_point = None, None  # enable grabbing a new rect
        dragging = False
        prior_rect = None
    elif event is not None and event.endswith('+RIGHT+') and window == canvas_window:  # Righ click
        window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
    elif event is not None and event.endswith('+MOTION+') and window == canvas_window:  # Righ click
        window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")
   
    elif event == 'Erase item' and window == canvas_window:
        window["-INFO-"].update(value=f"Right click erase at {values['-GRAPH-']}")
        if values['-GRAPH-'] != (None, None):
            drag_figures = graph.get_figures_at_location(values['-GRAPH-'])
            for figure in drag_figures:
                graph.delete_figure(figure)

    if event == '-CHANGE_COMPASS-':
        compass_offset += 1
        n_compass = len(compass_list)
        compass = compass_list[compass_offset%n_compass]
        try:
            prev_compass_coord, _ = graph.get_bounding_box(compass_canvas)
            graph.delete_figure(compass_canvas)
            del global_vars[str(compass_canvas)]
            compass_canvas = graph.draw_image(data=compass, location=prev_compass_coord)
            global_vars[str(compass_canvas)] = compass_canvas
        except:
            compass_canvas = graph.draw_image(data=compass, location=prev_compass_coord)
            global_vars[str(compass_canvas)] = compass_canvas
            prev_compass_coord, _ = graph.get_bounding_box(compass_canvas)
    if event == '-SAVE-':
        filename = sg.popup_get_file('Choose file (PNG, JPG) to save to\ne.g. image1.jpg or image1.png', save_as=True)
        
        if filename is None or filename == '':
            pass
        elif filename.endswith('.png') or filename.endswith('.jpg'):
            save_element_as_file(window['-GRAPH-'], filename)
        else:
            sg.popup('File name does not have an extension e.g. .png or .jpg',title="Error")
            pass

    if event in list(options.keys()):
        for e,attr_dict in options.items():
            attr_dict['bool'] = False
            window[e].update(image_data=options[e]['img_false'])
        options[event]['bool'] = True
        window[event].update(image_data=options[event]['img_true'])

    
    if event == 'Add Image' and values['-GEOREFERENCED_IMAGE_LIST-'] != '':
        img_file = values['-GEOREFERENCED_IMAGE_LIST-']
        line_number = int(re.sub(r'^.*?image_line_','',img_file)[:2])
        #get line number from fp and subset from predicted_bytes_list
        if 'Geotransformed_predicted' in img_file:
            
            img_base64 = pred_bytes_list[line_number]
        else:
            img_base64 = bytes_list[line_number]
        
        if len(global_vars['image_canvas_id'])>0:#if images have not been cleared
            prev_image_coord, _ = graph.get_bounding_box(global_vars['image_canvas_id'][0])#image_canvas_id[0])
            image_canvas = graph.draw_image(data=img_base64, location=prev_image_coord)
        else:
            image_canvas = graph.draw_image(data=img_base64, location=prev_image_coord)
            prev_image_coord, _ = graph.get_bounding_box(image_canvas)
       
        if 'Geotransformed_predicted' in img_file:
            predicted_image_canvas_id.append(image_canvas)
        else:
            image_canvas_id.append(image_canvas)
        
        if 'Geotransformed_predicted' in img_file:
            global_vars['image_canvas_id'] = image_canvas_id + predicted_image_canvas_id
        else:
            global_vars['image_canvas_id'] = image_canvas_id

    if event == 'Change background':
        background_counter += 1
        graph.update(background_color = background_dict[background_counter%background_len])
    if event == '-ADD_TEXT-':
        text_added = sg.popup_get_text('Enter text',title = None,default_text = "Text")
        text_canvas = graph.draw_text(text_added,font='Arial 10 normal',location=(g_size[0]//2,g_size[0]//2))
        global_vars[str(text_canvas)] = text_canvas
    if event == '-ADD_SPECTRAL-':
        if add_spectral_bool == True:
            tss_df_fp = join(fp_store,'Extracted_Spectral_Information',geotransformed_prefix+'_TSS_spectral_info.csv')
            try:
                wql_df = pd.read_csv(tss_df_fp)
            except Exception as E:
                sg.popup(f'{E}',title='Error')
                pass
            fig = plot_TSS_conc_image(wql_df)
            image = figure_to_image(fig)
            image = convert_to_bytes(image, (g_size[0],g_size[0]))
            window['-SPECTRAL_CANVAS-'].update(data=image)
            add_spectral_bool = False
        elif add_spectral_bool == False:
            window['-SPECTRAL_CANVAS-'].update(data=None)
            add_spectral_bool = True
        else:
            sg.popup(f'wql info not available: {tss_df_fp}')
            pass
   
    if event == '-ADD_WQL-':
        if add_wql_bool == True:
            radius = 5
            legend_radius = 8
            wql_graph_id = [] #list to store wql points id
            wql_legend = [] #list to store wql legend id
            # try:
            if len(tss_full_df.index) > 0:
                try:
                    current_image_coord, _ = graph.get_bounding_box(image_canvas_id[0])
                except:
                    sg.popup('No images present!',title='Warning')
                    pass
                initial_image_x = 0
                initial_image_y = int(img_resize_list[0].shape[0]*compass_scale)
                print(tss_full_df.head())
                for index,row in tss_full_df.iterrows():#tss_full_df.iterrows():
                   
                    x = row['x_general']*compass_scale
                    y = row['y_general']*compass_scale
                   
                    x_offset = x + (current_image_coord[0] - initial_image_x)
                    y_offset = y - (initial_image_y - current_image_coord[1])
                    
                    wql_id = graph.draw_circle((x_offset,y_offset),radius=radius,fill_color = row['hex'],line_color=row['hex'])
                    wql_graph_id.append(wql_id)
                
                #add legend
               
                legend_x = int(img_resize_list[0].shape[0]*compass_scale) - 120
                legend_y = 300
                line_space = 40
                leg_id = graph.draw_text('Legend',(legend_x+line_space,legend_y),font = 'arial 10 bold')
                wql_legend.append(leg_id)
               
                for i, (conc, h) in enumerate(wql_legend_content):
                    j = i+1
                    concentration = '{:.2f}'.format(conc)
                    leg_id = graph.draw_circle((legend_x,legend_y-line_space*j),radius=legend_radius,fill_color = h,line_color='black')
                    wql_legend.append(leg_id)
                    leg_id = graph.draw_text(concentration,(legend_x+line_space,legend_y-line_space*j),font = 'arial 8 normal')
                    wql_legend.append(leg_id)
                
                add_wql_bool = False
                global_vars['wql_graph_id'] = wql_graph_id
                global_vars['wql_legend'] = wql_legend
            else:
                sg.popup('No rows in wql df',title='Warning')
                pass
            
        elif add_wql_bool == False:
            for fig in wql_graph_id:
                graph.delete_figure(fig)
            for fig in wql_legend:
                graph.delete_figure(fig)
            add_wql_bool = True
            if 'wql_graph_id' in list(global_vars):
                del global_vars['wql_graph_id']
            if 'wql_legend' in list(global_vars):   
                del global_vars['wql_legend']
            
        else:
            sg.popup('No rows in wql df',title='Warning')
            pass
    
    if event == '-ADD_SCALE-':
        if add_scale == True:
            scale_legend = []
            segment_metres = 25
            segment = (img_resize_list[0].shape[1]/vc.dist*segment_metres)*compass_scale #ncols for 25meters
            scale_x = int(img_resize_list[0].shape[0]*compass_scale)//2
            scale_y = 50
            n_segments = 2
            line_width = 2
            down_tick_len = 20
            for s in range(n_segments):
                seg = graph.draw_line((scale_x+s*segment,scale_y),(scale_x+(s+1)*segment,scale_y),width=line_width)
                scale_legend.append(seg)
                down_ticks = graph.draw_line((scale_x+s*segment,scale_y),(scale_x+s*segment,scale_y-down_tick_len),width=line_width)
                scale_legend.append(down_ticks)
                scale_text = str(segment_metres*s)+' m'
                seg = graph.draw_text(scale_text,(scale_x+s*segment,scale_y+20),font = 'arial 8 normal')
                scale_legend.append(seg)
                if s == n_segments - 1:
                    s = n_segments
                    down_ticks = graph.draw_line((scale_x+s*segment,scale_y),(scale_x+s*segment,scale_y-down_tick_len),width=line_width)
                    scale_legend.append(down_ticks)
                    scale_text = str(segment_metres*s)+' m'
                    seg = graph.draw_text(scale_text,(scale_x+s*segment,scale_y+20),font = 'arial 8 normal')
                    scale_legend.append(seg)
            
            #add latlon grids
            #initial location of UL
            initial_image_x = 0
            
            initial_image_y = int(img_resize_list[0].shape[0]*compass_scale)
            UL_coord = (0,int(img_resize_list[0].shape[0]*compass_scale))
            UR_coord = (int(img_resize_list[0].shape[1]*compass_scale),int(img_resize_list[0].shape[0]*compass_scale))
            LL_coord = (0,0)
            LR_coord = (int(img_resize_list[0].shape[1]*compass_scale),0)
            scale_grid_ids = []
            #get current bounding box
            try:
                (UL_x,UL_y), (LR_x,LR_y) = graph.get_bounding_box(image_canvas_id[0])
            except:
                sg.popup('No images present!',title='Warning')
                pass
            #attributes
            n_segments = 2 #3 tick marks - start, middle, end
            line_width = 1
            tick_len = 20
            #width and height of the grid
            y_len = int(img_resize_list[0].shape[0]*compass_scale)
            x_len = int(img_resize_list[0].shape[1]*compass_scale)
    
            for s in range(n_segments+1): #0,1,2 for 3 tick marks
                f = s/n_segments #fraction of the length
                x = int(f*x_len) #horizontal distance from the left
                y = int(f*y_len) #vertical distance from the bottom
                #for vert ticks (horizontal lon axis), only x changes, y remains the same
                x_vert_tick = UL_x + x
                y_vert_tick = LR_y
                x_lon = f*(vc.right - vc.left) + vc.left
                vert_ticks = graph.draw_line((x_vert_tick,y_vert_tick),(x_vert_tick,y_vert_tick-tick_len),width=line_width)
                scale_grid_ids.append(vert_ticks)
                lon_str = deg_to_dms(x_lon, type='lon')
                lon_text = graph.draw_text(lon_str,(x_vert_tick,y_vert_tick-tick_len-10),font = 'arial 6 normal')
                scale_grid_ids.append(lon_text)
                #for horizontal ticks (vertical lat axis), only y changes, x remains the same
                x_hor_tick = UL_x
                y_hor_tick = LR_y + y
                y_lat = f*(vc.top - vc.bottom) + vc.bottom
                hor_ticks = graph.draw_line((x_hor_tick,y_hor_tick),(x_hor_tick-tick_len,y_hor_tick),width=line_width)
                scale_grid_ids.append(hor_ticks)
                lat_str = deg_to_dms(y_lat, type='lat')
                lat_text = graph.draw_text(lat_str,(x_hor_tick-tick_len-45,y_hor_tick),font = 'arial 6 normal')
                scale_grid_ids.append(lat_text)
            add_scale = False
            global_vars['scale_legend'] = scale_legend
            global_vars['scale_grid_ids'] = scale_grid_ids

        else:
            for fig in scale_legend:
                graph.delete_figure(fig)
            for fig in scale_grid_ids:
                graph.delete_figure(fig)
            add_scale = True
            if 'scale_legend' in list(global_vars):
                del global_vars['scale_legend']
            if 'scale_grid_ids' in list(global_vars):   
                del global_vars['scale_grid_ids']
    
    
    if event == '-ADD_PREDICTED-':
        if add_predicted == True:
            canvas_window['-GEOREFERENCED_IMAGE_LIST-'].update(values=vc.prediction_fp_list)
            
            if 'image_canvas_id' in global_vars.keys():
                prev_image_coord, _ = graph.get_bounding_box(image_canvas_id[0])
               
            
            predicted_image_canvas_id = []
            for b in pred_bytes_list:
                image_canvas = graph.draw_image(data=b, location=prev_image_coord) #if doesnt work change to n_cols
                predicted_image_canvas_id.append(image_canvas)
            
            image_canvas_id = image_canvas_id + predicted_image_canvas_id
            global_vars['image_canvas_id'] = image_canvas_id #update global vars with the appended predicted so that any move, delete will be consistent with above methods
            
            if tss_full_df is not None:
                model_performance_plot = vc.plot_model_performance()
                image = figure_to_image(model_performance_plot)
                image = convert_to_bytes(image, (g_size[0],g_size[0]))
                window['-SPECTRAL_CANVAS-'].update(data=image)
            add_predicted = False

        elif add_predicted == False:
            canvas_window['-GEOREFERENCED_IMAGE_LIST-'].update(values=vc.rgb_fp_list)
            for fig in predicted_image_canvas_id:
                graph.delete_figure(fig)
            
            if 'image_canvas_id' in global_vars.keys():
                image_canvas_id = [i for i in global_vars['image_canvas_id'] if i not in predicted_image_canvas_id]
                global_vars['image_canvas_id'] = image_canvas_id
                predicted_image_canvas_id = []
            #else there's nothing to remove
            window['-SPECTRAL_CANVAS-'].update(data=None)
            add_predicted = True
            
        else:
            sg.popup('Predicted images cannot be added, have the predicted images been created?',title="Error")
            pass
    
    if event == "-CORRECTION-":
        #enter the list of required info here
        required_info = {'image folder path':values['-IMAGE_FOLDER_FILEPATH-'],'height':values['-HEIGHT-'],'gps indices file path':values['-GPS_INDEX_TXT-']}
        
        if '' in required_info.values():
            missing_info_list = [k for k,v in required_info.items() if v == '']
            sg.popup('These elements are missing: {}\nInput these information, then click on \"Real-time image alignment correction\" button to continue!'.format(missing_info_list),title='Missing required elements!')
            pass
        else:
            try:
                with open(values['-GPS_INDEX_TXT-'], "r") as f:
                    gps_indices = f.readlines()
                gps_indices = [int(i.replace('\n','')) for i in gps_indices]
                gps_indices.sort()
                print(f'gps indices: {gps_indices}')
            except Exception as E:
                sg.popup(f"Error {E}",title="Error")
                pass
            gps_df = import_gps(values['-IMAGE_FOLDER_FILEPATH-'])
            unique_gps_df = get_unique_df(gps_df)

            if values['-LINE_START-']=='' and values['-LINE_END-']=='':
                trimmed_gps_indices = gps_indices
            elif values['-LINE_END-'] == '':
                line_start = int(values['-LINE_START-'])
                line_end = len(gps_indices)//2-1
                trimmed_gps_indices = gps_indices[int(line_start*2):int(line_end*2+2)]
            else:
                line_start = int(values['-LINE_START-'])
                line_end = int(values['-LINE_END-'])
                trimmed_gps_indices = gps_indices[int(line_start*2):int(line_end*2+2)]

            extended_rgb = ExtendedRGB(values['-IMAGE_FOLDER_FILEPATH-'],trimmed_gps_indices,int(values['-HEIGHT-']), unique_gps_df)
            general_dict,datetime_list = extended_rgb.main()
            live_correction = LiveCorrection(general_dict,datetime_list,time_delay=int(values["-SLIDER-"]),scale=20)
            
            img_resize_list, correction_bytes_list,bckgrnd_attr,bbox_list = live_correction.main()
            nrows, ncols, _ = img_resize_list[0].shape
            n_lines = len(img_resize_list)
            # print(f'nrows: {nrows} ncols: {ncols}')
            scale = 20
            canvas_scale = 1.5
            compass_scale = 2
            if nrows > ncols:
                g_size = (int(nrows),int(nrows))
            else:
                g_size = (int(ncols),int(ncols))
            correction_canvas_window = get_correction_canvas_window(g_size,compass_scale,n_lines)
            correction_graph = correction_canvas_window["-CORRECTION_GRAPH-"]
            correction_canvas_window['-SLIDER_CORRECTION-'].update(int(values["-SLIDER-"]))
            correction_image_canvas_id = {} #where keys are line_numbers, and values are the ID corresponding to canvas ID
            for i,b in enumerate(correction_bytes_list): #where i is the line_number
                image_canvas = correction_graph.draw_image(data=b, location=(0,int(g_size[0]*compass_scale))) #if doesnt work change to n_cols
                correction_image_canvas_id[i] = b
            
            corr_coeff_list,p_value = calculate_correlation_overlap(img_resize_list)
            corr_coeff_avg = np.nanmean(corr_coeff_list) #compute mean ignoring NaN value
            correction_canvas_window["-CORRECTION_INFO-"].update(value="Average correlation: {:.4f}".format(corr_coeff_avg))

    elif event == "-CORRECTION_GRAPH-":  # if there's a "Graph" event, then it's a mouse
        correction_graph.grab_anywhere_exclude()
    if event == "-SLIDER_CORRECTION-":
        """
        correct all image lines simultaneously using the same time delay
        """
        live_correction = LiveCorrection(general_dict,datetime_list,time_delay=int(values["-SLIDER_CORRECTION-"]),scale=20)
        img_resize_list, correction_bytes_list,bckgrnd_attr,bbox_list = live_correction.main()
        corr_coeff_list,_ = calculate_correlation_overlap(img_resize_list)
        corr_coeff_avg = np.nanmean(corr_coeff_list) #compute mean ignoring NaN value
        for i in range(n_lines): #update the slides on the side for better fine-tuning
            window["-S_{}-".format(i)].update(value=int(values["-SLIDER_CORRECTION-"]))
        window["-CORRECTION_INFO-"].update(value="Average correlation: {:.4f}".format(corr_coeff_avg))
        # erase previous images
        
        for b in correction_image_canvas_id.values():
            correction_graph.delete_figure(b)
        #add newly corrected images
        correction_image_canvas_id = {}
        for i,b in enumerate(correction_bytes_list):
            image_canvas = correction_graph.draw_image(data=b, location=(0,int(g_size[0]*compass_scale))) #if doesnt work change to n_cols
            correction_image_canvas_id[i] = b

    if event is not None and event[:3] == "-S_":
        """
        correct individual image lines
        """
        slider_number = int(event.replace('-','').replace('S_',''))
        s,ms = divmod(int(values['-S_{}-'.format(slider_number)]),1000)
        img_resize,correction_bytes = live_correction.correct_individual_lines(bbox_list,bckgrnd_attr,seconds_delay=s,milliseconds_delay=ms,line_number=slider_number)
        correction_graph.delete_figure(correction_image_canvas_id[slider_number])
        correction_graph.draw_image(data=correction_bytes, location=(0,int(g_size[0]*compass_scale)))

        #calculate correlation between adjacent lines only
        img_resize_list[slider_number] = img_resize
        if slider_number == n_lines - 1: #if it's the last line
            corr_coeff_list,_ = calculate_correlation_overlap(img_resize_list[slider_number-1:])
        elif slider_number == 0:
            corr_coeff_list,_ = calculate_correlation_overlap(img_resize_list[:2])
        else:
            corr_coeff_list,_ = calculate_correlation_overlap(img_resize_list[slider_number-1:slider_number+2])
        corr_coeff_str = ['{:.4f}'.format(c) for c in corr_coeff_list]
        window["-CORRECTION_INFO-"].update(value=f"Modifying line {slider_number}: Correlations: {corr_coeff_str}")

    if event == "-SAVE_CORRECTION-":
        """
        save corrected indices
        """
        try:
            corrected_indices_fp = sg.popup_get_folder("Folder to save corrected indices")
            live_correction.save_corrected_indices(corrected_indices_fp)
            sg.popup("corrected_indices.json saved in {}".format(corrected_indices_fp),title="Save successful")
        except Exception as E:
            sg.popup("Corrected file not saved yet",title="Save unsuccessful")
            pass
    
    if event == '-CORRECTED_IMG_INDICES-':
        corrected_indices_fp = sg.popup_get_file("Upload corrected_indices file")
        try:
            with open(corrected_indices_fp, "r") as read_file:
                corrected_indices = json.load(read_file)
            corrected_indices = [(i['start'],i['stop']) for i in corrected_indices]

            sg.popup("corrected_indices.json file successfully loaded!")
        except Exception as E:
            sg.popup(f"Invalid corrected_indices.json file\n{E}",title="Error")
            corrected_indices = None
            pass

    if event == "-GLINT-":
        lock = "glint"
        plt.clf()
        fig, ax = plt.subplots()
        DPI = fig.get_dpi()
        # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
        fig.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
    # -------------------------------
        current_fp = rgb_fp[img_counter%n_imges]
        img = Image.open(current_fp)
        img_line = int(rgb_fp[img_counter%n_imges].split('line_')[1][:2])
        ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
        im = ax.imshow(img)
        #draw lines
        print(f'xs_glint: {linebuilder.xs_glint}')
        print(f'bbox_glint: {linebuilder.img_bbox_glint}')

        line_glint, = ax.plot(0,0,"o",c="r")
        line_nonglint, = ax.plot(0,0,"o",c="purple")
        rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        r_glint = ax.add_patch(rect_glint)
        rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
        r_nonglint = ax.add_patch(rect_nonglint)
        figure_canvas_agg = draw_figure_w_toolbar(canvas=sgc_canvas_window['fig_cv'].TKCanvas,fig=fig, canvas_toolbar=sgc_canvas_window['controls_cv'].TKCanvas)
        linebuilder = LineBuilder(xs_glint=linebuilder.xs_glint,ys_glint=linebuilder.ys_glint,xs_nonglint=linebuilder.xs_nonglint,ys_nonglint=linebuilder.ys_nonglint,\
            line_glint=line_glint,line_nonglint=line_nonglint,r_glint=r_glint,r_nonglint=r_nonglint,\
            img_line_glint=linebuilder.img_line_glint,img_line_nonglint=linebuilder.img_line_nonglint,\
            img_bbox_glint=linebuilder.img_bbox_glint,img_bbox_nonglint=linebuilder.img_bbox_nonglint,\
            canvas=figure_canvas_agg,lock=lock,current_fp = current_fp)

    if event == "-NON_GLINT-":
        lock = "nonglint"
        plt.clf()
        fig, ax = plt.subplots()
        DPI = fig.get_dpi()
        # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
        fig.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
    # -------------------------------
        current_fp = rgb_fp[img_counter%n_imges]
        img = Image.open(current_fp)
        img_line = int(rgb_fp[img_counter%n_imges].split('line_')[1][:2])
        ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
        im = ax.imshow(img)
        #draw lines
        print(f'xs_nonglint: {linebuilder.xs_nonglint}')
        print(f'bbox_nonglint: {linebuilder.img_bbox_nonglint}')

        line_glint, = ax.plot(0,0,"o",c="r")
        line_nonglint, = ax.plot(0,0,"o",c="purple")
        rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        r_glint = ax.add_patch(rect_glint)
        rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
        r_nonglint = ax.add_patch(rect_nonglint)
        figure_canvas_agg = draw_figure_w_toolbar(canvas=sgc_canvas_window['fig_cv'].TKCanvas,fig=fig, canvas_toolbar=sgc_canvas_window['controls_cv'].TKCanvas)
        linebuilder = LineBuilder(xs_glint=linebuilder.xs_glint,ys_glint=linebuilder.ys_glint,xs_nonglint=linebuilder.xs_nonglint,ys_nonglint=linebuilder.ys_nonglint,\
            line_glint=line_glint,line_nonglint=line_nonglint,r_glint=r_glint,r_nonglint=r_nonglint,\
            img_line_glint=linebuilder.img_line_glint,img_line_nonglint=linebuilder.img_line_nonglint,\
            img_bbox_glint=linebuilder.img_bbox_glint,img_bbox_nonglint=linebuilder.img_bbox_nonglint,\
            canvas=figure_canvas_agg,lock=lock,current_fp = current_fp)

    if event == "-NEXT_IMAGE_SGC-":
        img_counter += 1
        print(f'img_counter: {img_counter}')
        print(f'linebuilder.img_line_glint: {linebuilder.img_line_glint}')
        print(f'linebuilder.img_line_nonglint: {linebuilder.img_line_nonglint}')
        plt.clf()
        fig, ax = plt.subplots()
        DPI = fig.get_dpi()
        # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
        fig.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
    # -------------------------------
        current_fp = rgb_fp[img_counter%n_imges]
        img = Image.open(current_fp)
        img_line = int(rgb_fp[img_counter%n_imges].split('line_')[1][:2])
        ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
        im = ax.imshow(img)
        #draw lines
        print(f'xs_nonglint: {linebuilder.xs_nonglint}')
        print(f'bbox_nonglint: {linebuilder.img_bbox_nonglint}')
        #reset points and boxes when changing image
        line_glint, = ax.plot(0,0,"o",c="r")
        line_nonglint, = ax.plot(0,0,"o",c="purple")
        rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        r_glint = ax.add_patch(rect_glint)
        rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
        r_nonglint = ax.add_patch(rect_nonglint)

        figure_canvas_agg = draw_figure_w_toolbar(canvas=sgc_canvas_window['fig_cv'].TKCanvas,fig=fig, canvas_toolbar=sgc_canvas_window['controls_cv'].TKCanvas)
        linebuilder = LineBuilder(xs_glint=linebuilder.xs_glint,ys_glint=linebuilder.ys_glint,xs_nonglint=linebuilder.xs_nonglint,ys_nonglint=linebuilder.ys_nonglint,\
            line_glint=line_glint,line_nonglint=line_nonglint,r_glint=r_glint,r_nonglint=r_nonglint,\
            img_line_glint=linebuilder.img_line_glint,img_line_nonglint=linebuilder.img_line_nonglint,\
            img_bbox_glint=linebuilder.img_bbox_glint,img_bbox_nonglint=linebuilder.img_bbox_nonglint,\
            canvas=figure_canvas_agg,lock=lock,current_fp = current_fp)

    if event == "-RESET_SGC-":
        reset_sgc(linebuilder)

    if event == "-SAVE_SGC-":
        #if sgc json file has been saved, next time dont need to select glint areas again, just click proceed
        line_glint = int(linebuilder.img_line_glint.split('line_')[1][:2]) #get the current image line
        line_nonglint = int(linebuilder.img_line_nonglint.split('line_')[1][:2]) #get the current image line
        bboxes = {'glint':{'line':line_glint,'fp':linebuilder.img_line_glint,'bbox':linebuilder.img_bbox_glint},\
            'non_glint':{'line':line_nonglint,'fp':linebuilder.img_line_nonglint,'bbox':linebuilder.img_bbox_nonglint}}
        with open(join(fp_store,'sunglint_correction_{}.txt'.format(config_file['-PREFIX-'])),'w') as cf:
            json.dump(bboxes,cf)
        sg.popup("sunglint_correction_{}.txt saved in {}".format(config_file['-PREFIX-'],fp_store),title="Save successful")

    if event == '-SELECT_DATE-':
        month, day, year = sg.popup_get_date()
        dt = (str(year),str(month).zfill(2),str(day).zfill(2))
        dt = '-'.join(dt)
        update_dt = values['-DATE_LIST-'] + ';' + dt
        window['-DATE_LIST-'].update(update_dt)
    
    if event == '-FETCH_ENV_DATE-':
        fp_store = sg.popup_get_folder("Folder to store environmental data")
        date_list = [i.strip() for i in values['-DATE_LIST-'].split(';')]
        n_dt = len(date_list)
        start_time_list = [i.strip() for i in values['-START_TIME_LIST-'].split(';')]
        end_time_list = [i.strip() for i in values['-END_TIME_LIST-'].split(';')]
        
        env_keys = {'-WIND_DIRECTION_CHECKBOX-':'wind-direction','-WIND_SPEED_CHECKBOX-':'wind-speed',\
            '-AIR_TEMPERATURE_CHECKBOX-':'air-temperature','-RELATIVE_HUMIDITY_CHECKBOX-':'relative-humidity'}
        env_params = []
        for k,v in env_keys.items():
            if values[k] is True:
                env_params.append(v)

        if values['-ENV_LOCATION-'] == '':
            sg.popup('Location not selected!',title="Warning")
            pass
        if fp_store == '' or fp_store is None:
            fp_store = None
            pass
        if len(start_time_list) != len(end_time_list):
            sg.popup('Length of start time list is not equal to length of end time list',title="Warning")
            pass
        elif len(start_time_list) != n_dt or len(end_time_list) != n_dt:
            sg.popup('Length of start time list or end time list is not equal to length of date time list',title="Warning")
            pass
        elif values['-START_TIME_LIST-'] == '' and values['-END_TIME_QQLIST-'] == '':
            start_time_list = ['00-00-01'] * n_dt
            end_time_list = ['23-59-59'] * n_dt
        
        try:
            sg.popup("Fetching data...\nUpon completion, there will be another popup.")
            date_time_dict = [{'start':convert_string_to_dt(dt,st),\
                'end':convert_string_to_dt(dt,et)}\
                    for dt,st,et in zip(date_list,start_time_list,end_time_list)]
            location_dict = get_env_locations()
            location_id = [id for id,name in location_dict.items() if name == values['-ENV_LOCATION-']][0]
            organise_data = create_nested_API_response(date_time_dict,env_params)
            organise_data = env_API_call(organise_data)
            organise_data = clean_env_API_response(location_id,organise_data)
            create_env_df(organise_data,location_id,fp_store)
            sg.popup('Environmental parameters fetched and saved in {}'.format(fp_store))
        except Exception as E:
            sg.popup(f"{E}",title="Error")
            pass

    if event == '-PLOT_ENV-':
        fp_env = sg.popup_get_file("File of retrieved environmental data")
        try:
            env_plot = plot_env_df(fp_env)
        except Exception as E:
            sg.popup(f"{E}\nFile cannot be plotted",title="Error")
            pass
        



            


        

window.close()
