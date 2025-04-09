//script that acquires data from eeg_headset, and stores/constantly updates a local vairable for signal processing use.
//this will be similar to the gNautilus demo, but instead of storing to a data.bin, we will constantly upload
//a variable scanWindow[]

#include "stdafx.h"
#include <Windows.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>

#include <cstdlib> 
#include <thread>

#include <GDSClientAPI_gNautilus.h>
#include <csignal>

#define SHARED_MEMORY_NAME L"Local\\GestureSharedMemory"  // Name of shared memory
#define SHARED_MEMORY_SIZE 256  // Buffer size

#define PYTHON_SCRIPT "\"C:\\Program Files\\Python310\\python.exe\"" 
#define SCRIPT_PATH "\"../hand-gesture-recognition-mediapipe/app.py\""

#define CONSOLE_ESCAPE_CODE_CLEAR_TO_THE_LEFT "\0"
#define CONSOLE_ESCAPE_CODE_CARRIAGE_RETURN "\r"

// Specifications for the data acquisition.
//-------------------------------------------------------------------------------------
float scanWindow[2500] = {0};  // This will hold the rolling 2s buffer
#define SAMPLE_RATE 250 // [Hz]
#define DURATION_DAQ 180 // [s]
#define DATA_READY_TRESHOLD_MS 30 // [ms]

// Definition of network specific stuff.
//-----------------------------------------------------------	--------------------------
#define HOST_IP "192.168.1.89" // Default address is the loopback address, else the ip of the computer running GDS.
#define HOST_PORT 50223     // The default port of GDS is 50223.
#define LOCAL_IP "192.168.1.76"// Default address is the loppback address, else the ip of the client machine.
#define LOCAL_PORT 50224    // Any free port on the local machine.

// Function declarations.
//-------------------------------------------------------------------------------------
void setup_config( GDS_GNAUTILUS_CONFIGURATION* config, unsigned int sample_rate );
void show_config( GDS_GNAUTILUS_CONFIGURATION* config );

void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData);
void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void*);
void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData);

// Global events plus timeout.
//-------------------------------------------------------------------------------------
#define SYSTEM_EVENT_TIMEOUT 5000 // [ms]
HANDLE glb_event_handle;

// Python Hand Gesture Script
//-------------------------------------------------------------------------------------
PROCESS_INFORMATION pi;

std::wstring stringToWString(const std::string& str) {
	return std::wstring(str.begin(), str.end());
}


void startPythonScript() {
	//std::string command = std::string(PYTHON_SCRIPT) + " " + SCRIPT_PATH;
	STARTUPINFO si = { sizeof(si) };

	std::string command = std::string("python ") + SCRIPT_PATH;
	// Convert to wide string (LPWSTR)
	std::wstring wcommand = stringToWString(command);
	LPWSTR lpwCommand = &wcommand[0];

	if (!CreateProcessW(NULL, lpwCommand, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
		std::cerr << "Failed to start Python script!" << std::endl;
	}
	else {
		std::cout << "Python script started successfully.\n";
	}
}

void stopPythonScript() {
	std::cout << "Stopping Python script...\n";
	TerminateProcess(pi.hProcess, 0);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
}

void handleExit(int signum) {
	stopPythonScript();

	exit(signum);
}
HANDLE createSharedMemory() {
	SECURITY_ATTRIBUTES sa;
	PSECURITY_DESCRIPTOR pSD = (PSECURITY_DESCRIPTOR)LocalAlloc(LPTR, SECURITY_DESCRIPTOR_MIN_LENGTH);

	if (!pSD) {
		std::cerr << "Failed to allocate security descriptor!" << std::endl;
		return NULL;
	}

	if (!InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION)) {
		std::cerr << "Failed to initialize security descriptor!" << std::endl;
		LocalFree(pSD);
		return NULL;
	}

	// Set DACL to NULL, allowing access to all users (needed for Python)
	if (!SetSecurityDescriptorDacl(pSD, TRUE, NULL, FALSE)) {
		std::cerr << "Failed to set security descriptor DACL!" << std::endl;
		LocalFree(pSD);
		return NULL;
	}

	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = pSD;
	sa.bInheritHandle = FALSE;

	std::string sharedMemoryName = "Local\\GestureSharedMemory";  // Narrow string
	std::wstring wSharedMemoryName = stringToWString(sharedMemoryName);  // Convert to wide string

	HANDLE hMapFile = CreateFileMappingW(
		INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, SHARED_MEMORY_SIZE, wSharedMemoryName.c_str());

	LocalFree(pSD);  // Free security descriptor
	return hMapFile;
}

//-------------------------------------------------------------------------------------
// main. Program entry point.
//-------------------------------------------------------------------------------------
int _tmain()
{
	bool stopBool = false;

	std::cout << "Symbionics right hand program start:" << std::endl << std::endl;

	signal(SIGINT, handleExit);  // Handle Ctrl+C (SIGINT)
	signal(SIGTERM, handleExit); // Handle termination signals

	HANDLE hMapFile = createSharedMemory();
	if (!hMapFile) {
		std::cerr << "Could not create shared memory!" << std::endl;
		return 1;
	}

	// Map shared memory to the process
	LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_SIZE);
	bool* requestData = reinterpret_cast<bool*>(pBuf);
	float* inputWindow = reinterpret_cast<float*>((char*)pBuf + sizeof(bool));

	if (!pBuf) {
		std::cerr << "Could not map view of file." << std::endl;
		CloseHandle(hMapFile);
		return 1;
	}

	// init global handles
	glb_event_handle = NULL;

	// Create the event which is triggerd when data should be processed.
	//-------------------------------------------------------------------------------------
	glb_event_handle = CreateEvent(NULL, false, false, NULL);
	if ( glb_event_handle == NULL )
	{
		CloseHandle(hMapFile);
		std::cerr << "ERROR: CreateEvent failed with error code " << GetLastError() << "." << std::endl;
		std::cin.get();
		return -1;
	}

	// Setup network addresses.
	//-------------------------------------------------------------------------------------
	static const std::string host_ip(HOST_IP);
	static const uint16_t host_port = HOST_PORT;
	static const std::string local_ip(LOCAL_IP);
	static const uint16_t local_port = LOCAL_PORT;
	GDS_ENDPOINT host_endpoint, local_endpoint;
	
	strcpy( host_endpoint.IpAddress, host_ip.c_str() );
	host_endpoint.Port = host_port;
	strcpy( local_endpoint.IpAddress, local_ip.c_str() );
	local_endpoint.Port = local_port;

	// Initialize the library.
	//-------------------------------------------------------------------------------------
	GDS_Initialize();

	// Get all connected devices.
	//-------------------------------------------------------------------------------------
	GDS_DEVICE_CONNECTION_INFO* connected_devices = NULL;
	size_t count_daq_units = 0;
	GDS_RESULT ret = GDS_GetConnectedDevices( host_endpoint, local_endpoint, &connected_devices, &count_daq_units );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_GetConnectedDevices: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		CloseHandle(hMapFile);
		return -1;
	}

	// Show all devices of interest (i.e. g.Nautilus).
	//-------------------------------------------------------------------------------------
	std::vector<std::string> device_list;
	size_t count_devices = 0;
	// The GDS_GetConnectedDevices returns the number of daq units. The first "for" loop
	// addresses the daq units.
	for ( size_t i = 0; i < count_daq_units; i++ )
	{
		// Each daq unit operates a different number of devices. The second "for" loop 
		// addresses the devices attached to the daq unit.
		for ( size_t j  = 0; j < connected_devices[i].ConnectedDevicesLength; j++ )
		{
			if ( connected_devices[i].ConnectedDevices[j].DeviceType == GDS_DEVICE_TYPE_GNAUTILUS )
			{
				if ( connected_devices[i].InUse == FALSE )
				{
					count_devices++;
					device_list.push_back( connected_devices[i].ConnectedDevices[j].Name ); 
					std::cout << "Device #" << count_devices << " (" << device_list[count_devices-1]  << " @ " << host_ip << ":" << host_port << ")" << std::endl;
				}
				else
					std::cout << "Device " <<	connected_devices[i].ConnectedDevices[j].Name  << " @ " << host_ip << ":" << host_port << " is not available " << std::endl;
			}
		}
	}
	if ( count_devices == 0 )
	{
		std::cout << "No device available, press <return> to exit" << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return 0;
	}
	else
	{
		// Free the list of found devices. This list is not needed any more.
		//-------------------------------------------------------------------------------------
		ret = GDS_FreeConnectedDevicesList( &connected_devices, count_daq_units);
		if ( ret.ErrorCode != GDS_ERROR_SUCCESS )
		{
			std::cerr << "ERROR on GDS_FreeConnectedDevicesList: " << ret.ErrorMessage << std::endl;
			CloseHandle(hMapFile);
			std::cin.get();
			return -1;
		}
	}

	// Read the input (i.e. the number of the device of interest) from the console.
	//-------------------------------------------------------------------------------------
	std::string str_selection;
	std::cout << "Select device #";
	std::getline( std::cin, str_selection );
	size_t idx_selection = atoi(str_selection.c_str())-1;
	if ( idx_selection > device_list.size()-1 )
	{
		std::cerr << "ERROR: The given device number is not valid" << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Try to connect to the device. This example deals with only a single device.
	//-------------------------------------------------------------------------------------
	char (*device)[DEVICE_NAME_LENGTH_MAX] = new char[1][DEVICE_NAME_LENGTH_MAX];
	std::strcpy( device[0], device_list[idx_selection].c_str() );
	GDS_HANDLE connectionHandle = 0;
	BOOL is_creator = FALSE;
	ret = GDS_Connect( host_endpoint, local_endpoint, device, 1, TRUE, &connectionHandle, &is_creator );
	delete[] device;
	device = NULL;
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_Connect: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Set the callback function which will be called if the server triggers a
	// daq error event.
	//-------------------------------------------------------------------------------------
	std::string usrData = device_list[idx_selection];
	ret = GDS_SetDataAcquisitionErrorCallback(connectionHandle, on_data_acquisition_error, (void*) &usrData);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetDataAcquisitionErrorCallback: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}
 
	// Set the callback function which will be called if the server dies or is going to
	// shutdown.
	//-------------------------------------------------------------------------------------
	ret = GDS_SetServerDiedCallback( connectionHandle, on_server_died_event, NULL );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on  GDS_SetServerDiedCallbackk: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Setup a configuration.
	//-------------------------------------------------------------------------------------
	GDS_GNAUTILUS_CONFIGURATION* cfg_nautilus = new GDS_GNAUTILUS_CONFIGURATION;
	setup_config( cfg_nautilus, SAMPLE_RATE );
	GDS_CONFIGURATION_BASE* cfg = new GDS_CONFIGURATION_BASE[1];
	cfg[0].DeviceInfo.DeviceType = GDS_DEVICE_TYPE_GNAUTILUS;
	strcpy( cfg[0].DeviceInfo.Name, device_list[idx_selection].c_str() );
	cfg[0].Configuration = cfg_nautilus;

	// Try to apply the configuration on the server.
	//-------------------------------------------------------------------------------------
	ret = GDS_SetConfiguration( connectionHandle, cfg, 1 );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetConfiguration " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Free the resources used by the config.
	//-------------------------------------------------------------------------------------
	delete cfg_nautilus;
	delete [] cfg;

	// Retrieve the configuration which has been applied to the server before.
	//-------------------------------------------------------------------------------------
	GDS_CONFIGURATION_BASE* cfg_received = NULL;
	size_t count_cfg_received = 0;
	ret = GDS_GetConfiguration( connectionHandle, &cfg_received, &count_cfg_received );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_GetConfiguration: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Write the retrieved config to the console.
	//-------------------------------------------------------------------------------------
	GDS_GNAUTILUS_CONFIGURATION* cfg_received_nautilus = (GDS_GNAUTILUS_CONFIGURATION*)(cfg_received[0].Configuration);
	std::cout << "Config of " << device_list[0] << ": " << std::endl;
	show_config( cfg_received_nautilus );

	// Set the data ready callback treshold.
	//-------------------------------------------------------------------------------------
	size_t data_ready_treshold_number_of_scans = std::max<size_t>(cfg_received_nautilus->SamplingRate * DATA_READY_TRESHOLD_MS/1000, cfg_received_nautilus->NumberOfScans );
	ret = GDS_SetDataReadyCallback(connectionHandle, on_data_ready_event, data_ready_treshold_number_of_scans, NULL);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetDataReadyCallback: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Free the resources used by the config.
	//-------------------------------------------------------------------------------------
	ret = GDS_FreeConfigurationList( &cfg_received, count_cfg_received );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_FreeConfigurationList: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Retrieve the buffer size for a single scan.
	//-------------------------------------------------------------------------------------
	size_t scan_count = 1;
	size_t channels_per_device_count = 0;
	size_t buffer_size_per_scan = 0;
	ret = GDS_GetDataInfo( connectionHandle, &scan_count, NULL, &channels_per_device_count, &buffer_size_per_scan );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on 1st GDS_GetDataInfo: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// The memory for the channels per device array is allocated here (based on the first call
	// of the method). GDS_GetDataInfo fills the array with the
	// available number of channels for each device. This is done for demo reasons.
	//-------------------------------------------------------------------------------------
	/*size_t* channels_per_device = new size_t[channels_per_device_count];
	ret = GDS_GetDataInfo( connectionHandle, &scan_count, channels_per_device, &channels_per_device_count, &buffer_size_per_scan );

	// We do not need the information because we know our own setup.
	//-------------------------------------------------------------------------------------
	delete[] channels_per_device;
	channels_per_device = NULL;
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on 2nd GDS_GetDataInfo: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}*/

	// Command the device to acquire data.
	//-------------------------------------------------------------------------------------
	ret = GDS_StartAcquisition( connectionHandle );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StartAcquisition(...): " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Stream the measurement data via the network.
	//-------------------------------------------------------------------------------------
	ret = GDS_StartStreaming( connectionHandle );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StartStreaming: " << ret.ErrorMessage << std::endl;
		CloseHandle(hMapFile);
		std::cin.get();
		return -1;
	}

	// Prepare a buffer and acquire the measurement data. Save the data to the file.
	//-------------------------------------------------------------------------------------
	size_t buffer_size_in_samples = SAMPLE_RATE * buffer_size_per_scan; //250 * ?
	uint64_t total_scans_to_acquire = DURATION_DAQ * SAMPLE_RATE;       // we don't want an end time but will set this to a long time in seconds
	uint64_t total_acquired_scans = 0;
    const size_t CHANNELS = 4;
    const size_t METADATA = 1;
    const size_t VALUES_PER_SCAN = CHANNELS + METADATA;  // = 5
    const size_t SCANS_PER_WINDOW = 250 * 2;             // = 500 total
    const size_t TOTAL_SAMPLES = SCANS_PER_WINDOW * VALUES_PER_SCAN;  // = 2500

	try
	{
		float* data_buffer = new float[buffer_size_in_samples];
		std::cout << "Buffer size in samples: " << buffer_size_in_samples << std::endl;
		std::cout << "Start acquiring measurement data for " << DURATION_DAQ << "s." << std::endl;
		while (true) //( total_acquired_scans < total_scans_to_acquire )
		{
			// wait until the server signals that the specified amount of data is available.
			DWORD dwWaitResult = WaitForSingleObject(glb_event_handle, SYSTEM_EVENT_TIMEOUT);
			if (dwWaitResult != WAIT_OBJECT_0)
				throw std::logic_error("Error: The data ready event hasn't been triggered within a reasonable time.");

			size_t scans_available = 0;
            //retrieve scans available from GDS server
			ret = GDS_GetData( connectionHandle, &scans_available, data_buffer, buffer_size_in_samples );
			if ( ret.ErrorCode )
			{
				std::cerr << "ERROR on GDS_GetData: " << ret.ErrorMessage << std::endl;
				break;
			}

			if ( scans_available > 0 )
			{
                //amount of samples we can copy into window right now
				size_t samples_to_copy = scans_available * VALUES_PER_SCAN; //5 values per scan

                //if the amount of samples is bigger than 2500, then we just copy the most recent 2500 samples
                if (samples_to_copy >= TOTAL_SAMPLES) {
                    std::memcpy(scanWindow, data_buffer + (samples_to_copy - TOTAL_SAMPLES), TOTAL_SAMPLES * sizeof(float));
                }
                //if the amount of samples is less than 2500, we just input the data, shifting the old data out
                else {
                    // Shift old data left
                    std::memmove(scanWindow, scanWindow + samples_to_copy, (TOTAL_SAMPLES - samples_to_copy) * sizeof(float));
                    // Copy new data to the end
                    std::memcpy(scanWindow + (TOTAL_SAMPLES - samples_to_copy), data_buffer, samples_to_copy * sizeof(float));
                }

                //check for flag from shawn here, to transfer this data to the NN classification
                if (*requestData == true) {
					std::memcpy(inputWindow, scanWindow, TOTAL_SAMPLES * sizeof(float));
					FlushViewOfFile(inputWindow, TOTAL_SAMPLES * sizeof(float));
					*requestData = false;
				}
				
			}
            //add a check here to end the infinite loop
            if (stopBool == 1){
                std::cout << "Ending Symbionics Program" << std::endl;
                break;
            }
		}
		std::cout << std::endl;

		// Free the buffer and close the file.
		//-------------------------------------------------------------------------------------
		CloseHandle(hMapFile);
		delete[] data_buffer;
		data_buffer = NULL;
	}
	catch(const std::logic_error & exc)
	{
		std::cerr << exc.what() << std::endl;
	}

	// Stop streaming data via the network.
	//-------------------------------------------------------------------------------------
	ret = GDS_StopStreaming(connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StopStreaming: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Command the device to stop acquiring data.
	//-------------------------------------------------------------------------------------
	ret = GDS_StopAcquisition(connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StopAcquisition: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Disconnect from the server.
	//-------------------------------------------------------------------------------------
	ret = GDS_Disconnect(&connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_Disconnect: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}
	
	// If the global handles exist then close them.
	//-------------------------------------------------------------------------------------
	if (glb_event_handle)
	{
		CloseHandle(glb_event_handle);
		glb_event_handle = NULL;
	}

	// Uninitialize the library.
	//-------------------------------------------------------------------------------------
	GDS_Uninitialize();

	std::cout << std::endl << "Press <return> to exit" << std::endl;
	std::cin.get();
	return 0;
}

// The method can be used to setup a simple config.
//-------------------------------------------------------------------------------------
void setup_config( GDS_GNAUTILUS_CONFIGURATION* config, unsigned int sampling_rate )
{
	config->Slave = FALSE;
	config->SamplingRate = sampling_rate;	
	config->NumberOfScans = 0;
	config->NetworkChannel = 0;
	config->DigitalIOs = FALSE;
	config->InputSignal = GDS_GNAUTILUS_INPUT_SIGNAL_ELECTRODE;
	config->AccelerationData = FALSE;
	config->BatteryLevel = FALSE;
	config->CAR = FALSE;
	config->Counter = TRUE;
	config->LinkQualityInformation = FALSE;
	config->ValidationIndicator = FALSE;
	config->NoiseReduction = FALSE;

	for (uint32_t j = 0; j < GDS_GNAUTILUS_CHANNELS_MAX; j++)
	{
		if (j < 4)
			config->Channels[j].Enabled = TRUE;
		else 
			config->Channels[j].Enabled = FALSE;
	
		config->Channels[j].Sensitivity = 187500.0;
		config->Channels[j].BandpassFilterIndex = 16;
		config->Channels[j].NotchFilterIndex = 1;
		config->Channels[j].BipolarChannel = -1;	
		config->Channels[j].UsedForNoiseReduction = FALSE;
		config->Channels[j].UsedForCar = FALSE;
	}
}

// The method can be used to write the config to a console.
//-------------------------------------------------------------------------------------
void show_config( GDS_GNAUTILUS_CONFIGURATION* config )
{
	std::cout << "Config.SamplingRate = " << config->SamplingRate << std::endl;
	std::cout << "Config.NumberOfScans = " << config->NumberOfScans << std::endl;
	std::cout << "Config.Slave = " << config->Slave << std::endl;
	std::cout << "Config.NetworkChannel = " << config->NetworkChannel << std::endl;
	std::cout << "Config.DigitalIOs = " << config->DigitalIOs << std::endl;
	std::cout << "Config.AccelerationData = " << config->AccelerationData << std::endl;
	std::cout << "Config.BatteryLevel = " << config->BatteryLevel << std::endl;
	std::cout << "Config.CAR = " << config->CAR << std::endl;
	std::cout << "Config.Counter = " << config->Counter << std::endl;
	std::cout << "Config.LinkQualityInformation = " << config->LinkQualityInformation << std::endl;
	std::cout << "Config.ValidationIndicator = " << config->ValidationIndicator << std::endl;
	std::cout << "Conifig.NoiseReduction = " << config->NoiseReduction << std::endl;

	std::cout << "Config.Channels[i].Enabled = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << i+1 << " ";
	std::cout << std::endl;
	
	std::cout << "Config.Channels[i].Sensitivity = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].Sensitivity << " ";
	std::cout << std::endl;
		
	std::cout << "Config.Channels[i].BandpassFilterIndex = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].BandpassFilterIndex << " ";
	std::cout << std::endl;

	std::cout << "Config.Channels[i].NotchFilterIndex = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].NotchFilterIndex << " ";
	std::cout << std::endl;

	std::cout << "Config.Channels[i].BipolarFilterIndex = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].BipolarChannel << " ";

	std::cout << "Config.Channels[i].UsedForNoiseReduction = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].UsedForNoiseReduction << " ";
	std::cout << std::endl;

	std::cout << "Config.Channels[i].UsedForCar = " << std::endl;
	for (uint32_t i = 0; i < GDS_GNAUTILUS_CHANNELS_MAX; i++)
		if ( config->Channels[i].Enabled )
			std::cout << config->Channels[i].UsedForCar << " ";
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
}

// The method is triggered periodically by the server as long as the specified
// amount of data is available.
//-------------------------------------------------------------------------------------
void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData)
{
	SetEvent(glb_event_handle);
}

// This method will be involved if the data acquisition struggles.
//-------------------------------------------------------------------------------------
void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData)
{
	std::string* device = (std::string*) usrData;
	std::clog << "------------------------------------" << std::endl;
	std::clog << "Handle			= " << connectionHandle << std::endl;
	std::clog << "Device			= " << *device << std::endl;
	std::clog << "Where			= onDataAcquisitionError" << std::endl;
	std::clog << "What			= " << result.ErrorMessage << std::endl;
	std::clog << "ErrorCode		= " << result.ErrorCode << std::endl;
	std::clog << std::endl;
}

// This method will be involved if the server dies.
//-------------------------------------------------------------------------------------
void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData)
{
	std::clog << "------------------------------------" << std::endl;
	std::clog << "Handle = " << connectionHandle << std::endl;
	std::clog << "What   = onServerDiedEvent" << std::endl; 
}
