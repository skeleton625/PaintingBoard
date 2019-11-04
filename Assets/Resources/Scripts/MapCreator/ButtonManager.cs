using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ButtonManager : MonoBehaviour
{
    /* 각 버튼 집단 오브젝트 객체 */
    [SerializeField]
    private GameObject mainButtonObject;
    [SerializeField]
    private GameObject treeButtonObject;
    [SerializeField]
    private GameObject stoneButtonObject;


    /* 메인 버튼에서 특정 버튼을 누를 때의 함수 */
    public void mainButtonClick()
    {
        mainButtonObject.SetActive(false);
        switch (transform.name.Split('_')[1])
        {
            case "Tree":
                treeButtonObject.SetActive(true);
                break;
            case "Water":
                mainButtonObject.SetActive(true);
                break;
            case "Stone":
                stoneButtonObject.SetActive(true);
                break;
        }
    }

    /* 오브젝트 관련 버튼을 누를 때의 함수 */
    public void objectsButtonClick()
    {
        string[] objectDatas = transform.name.Split('_');
        MapManager.selectObject(objectDatas[0], objectDatas[1]);
    }

    /* 뒤로가기 버튼을 누를 때의 함수 */
    public void BackwardButtonClick()
    {
        string buttonName = transform.name.Split('_')[1];
        switch(buttonName)
        {
            case "tBack":
                treeButtonObject.SetActive(false);
                break;
            case "sBack":
                stoneButtonObject.SetActive(false);
                break;
        }
        mainButtonObject.SetActive(true);
    }

    public void writeTextData()
    {
        ObjectManager.writeText();
    }

    public void readTextData()
    {
        MapManager.readText();
    }

    public void resetMapButtonClick()
    {
        MapManager.isReset = true;
    }
}
