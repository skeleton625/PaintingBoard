using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class MapManager : MonoBehaviour
{

    /* 선택 블록 오브젝트 */
    [SerializeField]
    private GameObject settingBlock;
    /* 비선택 표시 */
    [SerializeField]
    private Material unSelected;
    /* 선택 표시 */
    [SerializeField]
    private Material Possible;
    [SerializeField]
    private Material imPossible;
    /* 선택된 오브젝트 회전 속도 */
    [SerializeField]
    private float rotSpeed;

    /* 패널 실행 여부 */
    public static bool isPanelOn;
    /* 오브젝트 버튼 클릭 여부 */
    private static bool isObjectSelected;
    public static bool isReset;
    /* 버튼을 통해 선택된 오브젝트의 타입, 이름 */
    private static string selectedType;
    private static string selectedName;
    /* Terrain 지형 오브젝트 객체 */
    private static Terrain theTerrain;

    /* 맵 제작 오브젝트 번호 */
    private int craftingNum;
    /* 선택된 오브젝트 */
    private GameObject selectedObject;

    /* 이전에 선택된 블록, 선택된 블록 */
    private GameObject preSelectedBlock, curSelectedBlock;
    
    /* Raycast Hit 정보 */
    private RaycastHit hitInfo;
    private bool isCrafting;

    /* 사용할 오브젝트들을 메모리에 저장해 둠 */
    void Start()
    {
        theTerrain = GetComponent<Terrain>();

        /* 오브젝트를 둘 수 있는 블록 오브젝트 초기화 */
        preSelectedBlock = settingBlock;
        selectedObject = settingBlock;
        curSelectedBlock = settingBlock;

        /* 맵 데이터 초기화 */
        int terrainSize = ObjectManager.getTerrainSize();
        theTerrain.terrainData = generateTerrain(theTerrain.terrainData,0, 0,
            terrainSize, terrainSize, 4);
        generateSettingBlock();
    }

    void Update()
    {
        generateObject();
        createObjectToPoint();
        resetMap();
    }

    public static void selectObject(string _type, string _name)
    {
        selectedType = _type;
        selectedName = _name;
        isObjectSelected = true;
    }

    private void resetMap()
    {
        if(isReset)
        {
            /* Terrain 지형 초기화 */
            int terrainSize = ObjectManager.getTerrainSize();
            int blockCount = ObjectManager.getBlockCount();
            theTerrain.terrainData = generateTerrain(theTerrain.terrainData, 0, 0,
                terrainSize, terrainSize, 4);

            /* 생성된 모든 오브젝트 삭제 */
            for (int i = 0; i < blockCount; i++)
            {
                for (int j = 0; j < blockCount; j++)
                {
                    ObjectManager.deleteObject(i, j);
                    ObjectManager.setSelectedObject(i, j, 0);
                    GameObject Cube = GameObject.Find(i + "_" + j);
                    Cube.GetComponent<MeshRenderer>().material = unSelected;
                }
            }

            /* 선택된 오브젝트 제거 */
            craftingNum = 0;
            selectedObject.SetActive(false);
            /* Reset이 완료 되었으므로 false로 변환 */
            isReset = false;
        }
    }

    /* 오브젝트의 생성을 시작할 수 있도록 생성할 오브젝트 정의 함수 */
    private void generateObject()
    {
        if (isObjectSelected)
        {
            selectedObject.SetActive(false);
            isObjectSelected = false;
            switch (selectedType)
            {
                case "Tree":
                    selectedObject = ObjectManager.getTree(selectedName);
                    craftingNum = int.Parse(selectedName);
                    break;
                case "Stone":
                    selectedObject = ObjectManager.getStone(selectedName);
                    craftingNum = int.Parse(selectedName);
                    break;
                case "Water":
                    craftingNum = 14;
                    return;
            }
            selectedObject.SetActive(true);
        }
    }

    /* 마우스를 통해 원하는 자리에 오브젝트를 생성하는 함수 */
    private void createObjectToPoint()
    {
        
        if (craftingNum != 0 && !isPanelOn)
        {
            string[] selectedNum = curSelectedBlock.name.Split('_');
            int x = int.Parse(selectedNum[0]);
            int z = int.Parse(selectedNum[1]);

            /* 마우스가 위치한 선택 블록 표시 */
            if (Physics.Raycast(Camera.main.ScreenPointToRay(Input.mousePosition), 
                out hitInfo))
            {
                curSelectedBlock = hitInfo.transform.gameObject;
                /* 이전 선택 블록 투명화 */
                preSelectedBlock.GetComponent<MeshRenderer>().material = unSelected;
                /* 해당 위치에 오브젝트가 있을 경우 빨간색을 표시 */
                if (ObjectManager.getSelectedObject(x, z) != 0)
                    curSelectedBlock.GetComponent<MeshRenderer>().material = imPossible;
                /* 그렇지 않을 경우 연두색을 표시 */
                else
                    curSelectedBlock.GetComponent<MeshRenderer>().material = Possible;
                preSelectedBlock = curSelectedBlock;
            }

            /* W 키를 누를 경우, 선택한 오브젝트 회전 */
            if(Input.GetKey(KeyCode.W) && craftingNum != 14)
                selectedObject.transform.RotateAround( selectedObject.transform.position, Vector3.up, 
                    rotSpeed * Time.deltaTime);

            /* 왼쪽 마우스 버튼으로 선택한 오브젝트를 생성 */
            if (Input.GetMouseButtonDown(0) && ObjectManager.getSelectedObject(x, z) == 0)
            {
                selectedNum = curSelectedBlock.name.Split('_');
                x = int.Parse(selectedNum[0]);
                z = int.Parse(selectedNum[1]);
                isCrafting = true;
                /* 물 이외의 오브젝트일 경우 */
                if(craftingNum != 14)
                {
                    ObjectManager.setObject(
                        Instantiate(selectedObject, curSelectedBlock.transform.position, selectedObject.transform.rotation), 
                        x, z);
                    curSelectedBlock.GetComponent<MeshRenderer>().material = unSelected;
                }
                /* 물 오브젝트일 경우 */
                else
                    theTerrain.terrainData = generateTerrain(theTerrain.terrainData, x*4, z*4, 4, 4, 0);

                /* 생성한 오브젝트 정보 표시 */
                ObjectManager.setSelectedObject(x, z, craftingNum);
                isCrafting = false;
            }
            /* 마우스 오른쪽 키를 누를 경우, 오브젝트 삭제 */
            else if (Input.GetMouseButton(1))
            {
                /* 삭제하는 오브젝트가 물 오브젝트일 경우 */
                if(!ObjectManager.deleteObject(x, z))
                {
                    theTerrain.terrainData = 
                        generateTerrain(theTerrain.terrainData, x * 4, z * 4, 4, 4, 4);
                    ObjectManager.setSelectedObject(x, z, 0);
                }
            }
        }
    }

    /* 특정 위치의 Terrain 깊이 설정 함수 */
    private static TerrainData generateTerrain(TerrainData terrainData, int startX, int startZ, int endX, int endZ, int depth)
    {
        terrainData.SetHeights(startX, startZ, GenerateHeights(endX, endZ, depth));
        return terrainData;
    }

    /* 특정 영역에 높이를 설정해주는 함수 */
    private static float[,] GenerateHeights(int endX, int endZ, float depth)
    {
        float[,] heights = new float[endX, endZ];
        for (int x = 0; x < endX; x++)
        {
            for (int y = 0; y < endZ; y++)
                heights[x, y] = depth;
        }
        return heights;
    }

    /* 선택 블록 초기화 함수 */
    private void generateSettingBlock()
    {
        int blockCount = ObjectManager.getBlockCount();
        /* 프로그램 실행 시 이벤트 오브젝트 생성 */
        for(int i = 0; i < blockCount; i++)
        {
            for(int j = 0; j < blockCount; j++)
            {
                Vector3 createPos = new Vector3(i * 4+2, 4, j * 4+2);
                GameObject clone = Instantiate(settingBlock, createPos, Quaternion.identity);
                clone.name = i.ToString() + '_' + j.ToString();
            }
        }
    }

    public static void readText()
    {
        if (new FileInfo("map_data_rot.csv").Exists && new FileInfo("map_data_type.csv").Exists)
        {
            int terrainSize = ObjectManager.getTerrainSize();
            StreamReader data_type = new StreamReader("map_data_type.csv", false);
            StreamReader data_rot = new StreamReader("map_data_rot.csv", false);
            string[] line_type;
            string[] line_rot;

            for (int i = 0; i < terrainSize; i++)
            {
                GameObject clone = null;
                line_type = data_type.ReadLine().Replace(" ", "").Split(',');
                line_rot = data_rot.ReadLine().Replace(" ", "").Split(',');
                for (int j = 0; j < terrainSize-1; j++)
                {
                    int types = int.Parse(line_type[j]);

                    if (types == 0 || (i%4 != 0 || j%4 != 0))
                        continue;

                    int x = i / 4, z = j / 4;
                    ObjectManager.setSelectedObject(x, z, types);

                    if (types < 11)
                        clone = Instantiate(ObjectManager.getTree(line_type[j]),
                                        new Vector3(i+2, 4,j+2),
                                        Quaternion.EulerAngles(0f, float.Parse(line_rot[j]), 0f));
                    else if(types < 14)
                        clone = Instantiate(ObjectManager.getStone(line_type[j]),
                                        new Vector3(i+2, 4, j+2),
                                        Quaternion.EulerAngles(0f, float.Parse(line_rot[j]), 0f));
                    else if (types == 14)
                    {
                        theTerrain.terrainData =
                           generateTerrain(theTerrain.terrainData, x * 4, z * 4, 4, 4, 0);
                        continue;
                    }
                    clone.SetActive(true);
                    ObjectManager.setObject(clone, x, z);
                }
            }

            data_rot.Close();
            data_type.Close();
        }
    }
}


