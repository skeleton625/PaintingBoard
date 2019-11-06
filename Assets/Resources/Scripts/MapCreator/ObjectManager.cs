using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ObjectManager : MonoBehaviour
{
    /* 돌 오브젝트 프리팹 */
    [SerializeField]
    private GameObject[] Stones;
    [SerializeField]
    /* 나무 오브젝트 프리팹 */
    private GameObject[] Trees;
    [SerializeField]
    private int tSize;
    [SerializeField]
    private int bCount;
    /* 맵 오브젝트 크기 정보 */
    private static int terrainSize;
    /* 맵에 설치된 이벤트 블록 개수 */
    private static int blockCount;
    /* 해당 칸에 대한 정보 배열들 */
    private static int[,] isSelected;
    private static GameObject[,] createdObject;
    /* 오브젝트 데이터들 */
    private static Dictionary<string, GameObject> objectDictionary;

    // Start is called before the first frame update
    void Start()
    {
        terrainSize = tSize;
        blockCount = bCount;
        isSelected = new int[blockCount, blockCount];
        createdObject = new GameObject[blockCount, blockCount];
        /* 나무, 돌 오브젝트 사전 초기화 */
        objectDictionary = new Dictionary<string, GameObject>();
        string inputObject;
        for (int i = 0; i < Trees.Length; i++)
        {
            inputObject = Trees[i].name;
            objectDictionary.Add(inputObject, Trees[i]);
        }
        for (int i = 0; i < Stones.Length; i++)
        {
            inputObject = Stones[i].name;
            objectDictionary.Add(inputObject, Stones[i]);
        }
        for(int i = 0; i < blockCount; i++)
        {
            for (int j = 0; j < blockCount; j++)
                isSelected[i, j] = 0;
        }
    }

    public static GameObject getObject(string name)
    {
        return objectDictionary[name];
    }

    public static GameObject getObject(int x, int y)
    {
        return createdObject[x, y];
    }

    public static void setObject(GameObject mapObject, int x, int y)
    {
        createdObject[x, y] = mapObject;
    }

    public static int getTerrainSize()
    {
        return terrainSize;
    }

    public static int getBlockCount()
    {
        return blockCount;
    }

    public static int getSelectedObject(int x, int y)
    {
        return isSelected[x, y];
    }

    public static void setSelectedObject(int x, int y, int num)
    {
        isSelected[x, y] = num;
    }

    public static bool deleteObject(int x, int y)
    {
        /* 삭제 불가능 오브젝트 */
        if (createdObject[x, y] == null || isSelected[x, y] == -1)
            return false;

        /* 삭제 가능한 오브젝트면서 비어있지 않을 경우 */
        if (createdObject[x, y] != null)
        {
            Destroy(createdObject[x, y]);
            createdObject[x, y] = null;
            isSelected[x, y] = 0;
            return true;
        }
        /* 예기치 못한 경우 */
        return false;
    }

    /*
    public static void writeText()
    {
        StreamWriter dataType = new StreamWriter("map_data_type.csv", false);
        StreamWriter data_rot = new StreamWriter("map_data_rot.csv", false);
        string input_type;
        string input_rot;

        for (int i = 0; i < terrainSize; i++)
        {
            input_type = "";
            input_rot = "";
            for (int j = 0; j < terrainSize; j++)
            {
                if (i % 4 == 0 && j % 4 == 0)
                {
                    int blockX = i / 4, blockZ = j / 4;
                    input_type += isSelected[blockX, blockZ];
                    if (isSelected[blockX, blockZ] > 0 && isSelected[blockX, blockZ] < 14)
                        input_rot += createdObject[blockX, blockZ].transform.rotation.y;
                    else
                       input_rot += "0";
                }
                else
                {
                    input_type += "0";
                    input_rot += "0";
                }

                if(j < terrainSize - 1)
                {
                    input_type += ",";
                    input_rot += ",";
                }
            }
            dataType.WriteLine(input_type);
            data_rot.WriteLine(input_rot);
        }
        dataType.Close();
        data_rot.Close();
    }
    */
}
