using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BoardButtonManager : MonoBehaviour
{
    private BoardManager GameBoard;

    // Start is called before the first frame update
    void Start()
    {
        GameBoard = GameObject.Find("StaticObject").GetComponent<BoardManager>();
    }

    public void OnColorButtonClick()
    {
        int _selectNum = int.Parse(gameObject.name.Split('_')[1]);
        GameBoard.SetPixelColor(_selectNum);
    }

    public void OnIamgeWriteButtonClick()
    {
        GameBoard.CreateImageFiles();
    }
}
